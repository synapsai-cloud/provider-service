"""Deploy a caller-supplied image as a Modal Server.

Modal docs used here:

- Servers: https://modal.com/docs/guide/servers
- Existing images: https://modal.com/docs/guide/existing-images
- GPUs: https://modal.com/docs/reference/modal.gpu
- Proxy auth: https://modal.com/docs/guide/webhook-proxy-auth
- Stopping an app: ``modal.experimental.stop_app`` (same operation as ``modal app stop``)

This module is imported inside the deployed container only as a cloudpickled
class. Keep the server entrypoint on the standard library so the image
stays self-contained.
"""

from __future__ import annotations

import json
from typing import Any

from provider_service.providers.schema import (
    INSTANCE_STATUS_BUILDING,
    INSTANCE_STATUS_FAILED,
    INSTANCE_STATUS_READY,
    INSTANCE_STATUS_SLEEPING,
    INSTANCE_STATUS_STOPPED,
    ModelStatusSummary,
    ProviderInstance,
    assign_warm_by_min_instances,
    status_from_instance_statuses,
)
from provider_service.settings import (
    DEFAULT_LISTEN_PORT,
    MODAL_SERVER_NAME,
    InferenceBackendError,
    function_name_from_app,
    cuda_visible_devices_value,
    modal_app_name,
    modal_environment_name,
    modal_gpu,
    modal_server_unauthenticated,
    registry_credentials,
)

# Model load happens before the port opens. Modal kills the container if the
# port is not listening within this many seconds.
DEFAULT_STARTUP_TIMEOUT = 1800
DEFAULT_TARGET_CONCURRENCY = 4
_MIN_CONTAINERS: dict[str, int] = {}


class ModalBackendError(InferenceBackendError):
    pass


def _environment() -> str | None:
    return modal_environment_name()


def _startup_timeout() -> int:
    import os

    raw = (os.getenv("MODAL_STARTUP_TIMEOUT") or "").strip()
    if raw == "":
        return DEFAULT_STARTUP_TIMEOUT
    try:
        value = int(raw)
    except ValueError as exc:
        raise ModalBackendError("MODAL_STARTUP_TIMEOUT must be a positive integer") from exc
    if value <= 0:
        raise ModalBackendError("MODAL_STARTUP_TIMEOUT must be a positive integer")
    return value


def _registry_secret(image: str):
    """Secret for a private registry pull. Public images deploy without one."""
    import modal

    username, password = registry_credentials()
    if username == "" or password == "":
        return None
    if not str(image).startswith("ghcr.io/"):
        return None
    return modal.Secret.from_dict(
        {
            "REGISTRY_USERNAME": username,
            "REGISTRY_PASSWORD": password,
        }
    )


# vLLM's entrypoint is ``vllm serve``, which swallows the command Modal uses to
# read the Python version. The image already has ``/usr/local/bin/python``.
# Do not use ``add_python``: Modal then runs ``ln -s`` onto that path and the
# build fails because the file exists.
_MODAL_PYTHON_SETUP = [
    "ENTRYPOINT []",
    "CMD []",
    "RUN ln -sfn \"$(command -v python3)\" /usr/local/bin/python",
]


def _inference_image(image: str, *, prepare_python: bool = False):
    """``Image.from_registry`` of the tag the caller sent.

    ``prepare_python`` clears the image entrypoint so Modal can see its interpreter.
    """
    import modal

    kwargs: dict[str, Any] = {}
    if prepare_python:
        kwargs["setup_dockerfile_commands"] = list(_MODAL_PYTHON_SETUP)
    secret = _registry_secret(image)
    if secret is not None:
        kwargs["secret"] = secret
    return modal.Image.from_registry(image, **kwargs)


def build_inference_app(
    *,
    app_name: str,
    command: list[str],
    image: str,
    gpu: str,
    min_containers: int,
    max_containers: int,
    scaledown_window: int,
    hf_token: str | None = None,
    workdir: str | None = None,
    target_concurrency: float | None = DEFAULT_TARGET_CONCURRENCY,
    prepare_python: bool = False,
    gpu_count: int = 1,
    extra_env: dict[str, str] | None = None,
    port: int | None = None,
) -> Any:
    """Build a Modal App whose Server starts ``command``."""
    import modal

    listen = int(port or DEFAULT_LISTEN_PORT)
    app = modal.App(app_name)
    container_image = _inference_image(image, prepare_python=prepare_python)
    env = {
        "SYNAPSAI_SERVE_COMMAND": json.dumps(list(command)),
        "PORT": str(listen),
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices_value(gpu_count),
    }
    if workdir:
        env["SYNAPSAI_SERVE_CWD"] = workdir
    secrets = []
    secret_values: dict[str, str] = {}
    if hf_token:
        secret_values["HF_TOKEN"] = hf_token
    if extra_env:
        secret_values.update({key: value for key, value in extra_env.items() if value})
    if secret_values:
        secrets.append(modal.Secret.from_dict(secret_values))

    server_kwargs: dict[str, Any] = {
        "image": container_image,
        "gpu": gpu,
        "env": env,
        "secrets": secrets or None,
        "min_containers": min_containers,
        "max_containers": max_containers,
        "scaledown_window": scaledown_window,
        "startup_timeout": _startup_timeout(),
        "port": listen,
        "name": MODAL_SERVER_NAME,
        "unauthenticated": modal_server_unauthenticated(),
        "serialized": True,
        "include_source": False,
    }
    if target_concurrency is not None:
        server_kwargs["target_concurrency"] = target_concurrency

    @app.server(**server_kwargs)
    class Inference:
        @modal.enter()
        def start(self):
            import json as _json
            import os
            import subprocess

            command = _json.loads(os.environ["SYNAPSAI_SERVE_COMMAND"])
            cwd = os.environ.get("SYNAPSAI_SERVE_CWD") or None
            subprocess.Popen(command, cwd=cwd)

    return app


def _server(app_name: str):
    import modal

    return modal.Server.from_name(
        app_name,
        MODAL_SERVER_NAME,
        environment_name=_environment(),
    )


def deploy_inference_server(
    *,
    model_id: str,
    command: list[str],
    image: str,
    gpu_count: int = 1,
    vram_mb: int = 24000,
    min_instances: int = 0,
    max_instances: int = 1,
    worker_timeout: float | None = None,
    hf_token: str | None = None,
    workdir: str | None = None,
    prepare_python: bool = False,
    request_concurrency: int | None = None,
    extra_env: dict[str, str] | None = None,
    port: int | None = None,
) -> dict[str, Any]:
    """Deploy or update the Modal App for one model and return its URL."""
    if not modal_server_unauthenticated() and _proxy_configured() is False:
        raise ModalBackendError(
            "Modal Servers require a proxy token. Set MODAL_PROXY_TOKEN_ID and "
            "MODAL_PROXY_TOKEN_SECRET, or MODAL_SERVER_UNAUTHENTICATED=1 for a public endpoint. "
            "See https://modal.com/docs/guide/webhook-proxy-auth"
        )
    if not str(image or "").strip():
        raise ModalBackendError("image is required")
    app_name = modal_app_name(model_id)
    resolved_image = str(image).strip()
    gpu = modal_gpu(vram_mb, gpu_count)
    min_containers = max(0, int(min_instances))
    max_containers = max(min_containers, int(max_instances), 1)
    timeout = 300 if worker_timeout is None else max(0, int(float(worker_timeout)))
    # A fixed-size pool should not also autoscale. Scale-from-zero needs a target.
    concurrency = DEFAULT_TARGET_CONCURRENCY if request_concurrency is None else max(1, int(request_concurrency))
    target = None if min_containers == max_containers and min_containers >= 1 else concurrency
    app = build_inference_app(
        app_name=app_name,
        command=command,
        image=resolved_image,
        gpu=gpu,
        min_containers=min_containers,
        max_containers=max_containers,
        scaledown_window=timeout,
        hf_token=hf_token,
        extra_env=extra_env,
        workdir=workdir,
        target_concurrency=target,
        prepare_python=prepare_python,
        gpu_count=gpu_count,
        port=port,
    )
    app.deploy(environment_name=_environment())
    url = _server(app_name).get_url()
    if not url:
        raise ModalBackendError(f"Modal did not return a URL for {app_name}")
    _MIN_CONTAINERS[app_name] = min_containers
    return {
        "ok": True,
        "app_name": app_name,
        "url": str(url).rstrip("/"),
        "image": resolved_image,
        "gpu": gpu,
    }


def _proxy_configured() -> bool:
    from provider_service.settings import modal_proxy_bearer

    return modal_proxy_bearer() != ""


def lookup_server_url(model_id: str) -> str:
    app_name = modal_app_name(model_id)
    try:
        url = _server(app_name).get_url()
    except Exception as exc:
        raise ModalBackendError(f"Modal app {app_name} is not deployed") from exc
    if not url:
        raise ModalBackendError(f"Modal app {app_name} has no server URL")
    return str(url).rstrip("/")


def stop_inference_server(model_id: str) -> dict[str, Any]:
    """Stop the deployed App. Missing apps are treated as already deleted."""
    import modal

    app_name = modal_app_name(model_id)
    _MIN_CONTAINERS.pop(app_name, None)
    try:
        modal.experimental.stop_app(app_name, environment_name=_environment())
    except Exception as exc:
        text = str(exc).lower()
        not_found = exc.__class__.__name__ in {"NotFoundError", "NotFound"} or "not found" in text
        if not_found:
            return {"ok": True, "missing": True, "app_name": app_name}
        raise ModalBackendError(f"Failed to stop Modal app {app_name}: {exc}") from exc
    return {"ok": True, "app_name": app_name}


def update_server_autoscaler(
    model_id: str,
    *,
    min_instances: int,
    max_instances: int,
    worker_timeout: float | None = None,
) -> dict[str, Any]:
    app_name = modal_app_name(model_id)
    min_containers = max(0, int(min_instances))
    max_containers = max(min_containers, int(max_instances), 1)
    kwargs: dict[str, Any] = {
        "min_containers": min_containers,
        "max_containers": max_containers,
    }
    if worker_timeout is not None:
        kwargs["scaledown_window"] = max(0, int(float(worker_timeout)))
    if not (min_containers == max_containers and min_containers >= 1):
        kwargs["target_concurrency"] = DEFAULT_TARGET_CONCURRENCY
    _server(app_name).update_autoscaler(**kwargs)
    _MIN_CONTAINERS[app_name] = min_containers
    return {"ok": True, "app_name": app_name}


def _container_status(status: str) -> str:
    normalized = str(status or "").strip().upper()
    if normalized in {"RUNNING", "READY", "WORKING", "IDLE"}:
        return INSTANCE_STATUS_READY
    if normalized in {"PENDING", "SCHEDULED", "RESTORING", "RESUMING", "CREATING", "STARTING", "SLEEPING", "STANDBY"}:
        return INSTANCE_STATUS_SLEEPING
    if normalized in {"STOPPING", "TERMINATING", "TERMINATED", "STOPPED", "DELETED", ""}:
        return INSTANCE_STATUS_STOPPED
    if normalized in {"FAILED", "ERROR"}:
        return INSTANCE_STATUS_FAILED
    if normalized == "BUILDING":
        return INSTANCE_STATUS_BUILDING
    return INSTANCE_STATUS_SLEEPING


def _min_containers_for_app(app_name: str, app: Any = None) -> int:
    if app_name in _MIN_CONTAINERS:
        return _MIN_CONTAINERS[app_name]
    direct = getattr(app, "min_containers", None) if app is not None else None
    if direct is not None:
        try:
            value = max(0, int(direct))
            _MIN_CONTAINERS[app_name] = value
            return value
        except (TypeError, ValueError):
            pass
    try:
        settings = _server(app_name).autoscaler.settings()
        value = max(0, int(getattr(settings, "min_containers", 0) or 0))
        _MIN_CONTAINERS[app_name] = value
        return value
    except Exception:
        return 0


def normalize_server_instance(pod: dict[str, Any]) -> ProviderInstance:
    app_name = str(pod.get("app_name") or "").strip()
    funcname = str(pod.get("funcname") or "").strip()
    if not funcname and app_name:
        funcname = function_name_from_app(app_name)
    name = str(pod.get("name") or pod.get("id") or funcname)
    pod_id = str(pod.get("id") or name or funcname)
    raw_state = str(pod.get("state") or "Running").strip()
    raw_status = str(pod.get("status") or raw_state).strip()
    return ProviderInstance(
        id=pod_id,
        name=name,
        funcname=funcname,
        state=raw_state,
        status=_container_status(raw_status),
        warm=bool(pod.get("warm", False)),
        backend="modal",
    )


def summarize_server_instances(
    pods: list[dict[str, Any]],
    *,
    min_instances: int = 0,
) -> ModelStatusSummary:
    instances = [normalize_server_instance(pod) for pod in pods]
    instances = assign_warm_by_min_instances(instances, min_instances)
    status, active = status_from_instance_statuses([inst.status for inst in instances])
    serving = [inst for inst in instances if inst.status == INSTANCE_STATUS_READY]
    return ModelStatusSummary(
        status=status,
        active_instances=active,
        serving_instances=serving,
    )


def list_server_instances() -> list[dict[str, Any]]:
    """Running containers for SynapsAI apps in the standard provider format."""
    import modal

    try:
        apps = modal.experimental.list_deployed_apps(environment_name=_environment() or "")
    except TypeError:
        apps = modal.experimental.list_deployed_apps()
    rows: list[dict[str, Any]] = []
    for app in apps:
        app_name = str(getattr(app, "name", "") or "")
        if not app_name.startswith("synapsai-"):
            continue
        funcname = function_name_from_app(app_name)
        minimum = _min_containers_for_app(app_name, app)
        containers = _list_containers(app_name)
        if containers:
            container_ids = sorted(
                str(getattr(container, "container_id", "") or "")
                for container in containers
                if str(getattr(container, "container_id", "") or "") != ""
            )
            for idx, container_id in enumerate(container_ids):
                rows.append(
                    {
                        "id": container_id,
                        "name": container_id,
                        "funcname": funcname,
                        "state": "Running",
                        "status": INSTANCE_STATUS_READY,
                        "warm": idx < minimum,
                        "backend": "modal",
                        "app_name": app_name,
                    }
                )
            continue
        running = int(getattr(app, "containers", 0) or 0)
        if running <= 0:
            continue
        for idx in range(running):
            cid = app_name if idx == 0 else f"{app_name}-{idx + 1}"
            rows.append(
                {
                    "id": cid,
                    "name": cid,
                    "funcname": funcname,
                    "state": "Running",
                    "status": INSTANCE_STATUS_READY,
                    "warm": idx < minimum,
                    "backend": "modal",
                    "app_name": app_name,
                }
            )
    return rows


def _list_containers(app_name: str) -> list[Any]:
    try:
        return list(_server(app_name)._experimental_list_containers())
    except Exception:
        return []


def list_server_functions() -> list[dict[str, str]]:
    import modal

    try:
        apps = modal.experimental.list_deployed_apps(environment_name=_environment() or "")
    except TypeError:
        apps = modal.experimental.list_deployed_apps()
    rows = []
    for app in apps:
        app_name = str(getattr(app, "name", "") or "")
        if not app_name.startswith("synapsai-"):
            continue
        rows.append({"name": function_name_from_app(app_name), "app_name": app_name})
    return rows


def get_model_logs(model_id: str, *, tail: int = 200) -> str:
    """Recent Modal Server logs for this model, including container stdout."""
    app_name = modal_app_name(model_id)
    limit = max(1, int(tail))
    try:
        server = _server(app_name)
        chunks: list[str] = []
        for entry in server.logs.tail(limit):
            chunks.append(str(getattr(entry, "message", "") or ""))
            if len(chunks) >= limit:
                break
    except Exception as exc:
        raise ModalBackendError(f"Failed to read Modal logs for {app_name}: {exc}") from exc
    return "".join(chunks)


def stop_server_container(container_id: str) -> dict[str, Any]:
    """Stop one container. Same RPC as ``modal container stop --graceful``."""
    import asyncio

    import modal
    from modal.client import _Client
    from modal_proto import api_pb2

    target = str(container_id or "").strip()
    if target == "":
        raise ModalBackendError("container id is required")

    async def _stop() -> None:
        client = await _Client.from_env()
        await client._stub.ContainerStop(
            api_pb2.ContainerStopRequest(task_id=target, graceful=True)
        )

    try:
        asyncio.run(_stop())
    except RuntimeError:
        # Already inside a running loop: drive the coroutine on a fresh loop thread.
        import threading

        error: list[BaseException] = []

        def _run() -> None:
            try:
                asyncio.run(_stop())
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()
        if error:
            raise ModalBackendError(f"Failed to stop Modal container {target}: {error[0]}") from error[0]
    except Exception as exc:
        raise ModalBackendError(f"Failed to stop Modal container {target}: {exc}") from exc
    return {"ok": True, "id": target}

