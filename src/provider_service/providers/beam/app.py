"""Deploy a caller-supplied image on Beam.

Scale-to-zero and extra replicas are serverless Pods. Replicas that must stay
up (``min_instances``) run on a reserved on-demand machine.
``BEAM_ONDEMAND_ONLY=1`` puts every replica, including ``min_instances`` 0,
on a reserved on-demand machine.


Beam docs used here:

- Pods: https://docs.beam.cloud/v2/reference/py-sdk (``Pod``)
- GPU: https://docs.beam.cloud/v2/environment/gpu
- Images: https://docs.beam.cloud/v2/reference/py-sdk (``Image.from_registry``)
- Memory checkpoints: https://docs.beam.cloud/v2/reference/py-sdk (``checkpoint_enabled``)
- Snapshots: https://docs.beam.cloud/v2/sandbox/snapshots
- Auth: https://docs.beam.cloud/v2/endpoint/overview (``Authorization: Bearer``)

When checkpointing is enabled, Beam captures a memory snapshot after the
caller's readiness path returns 200, then stops that container. A serverless pod
does not start until the first request, so the checkpoint wait lives in
deployment verification: after ``/model/ready`` succeeds and before the
stress test. ``wait_for_checkpoint`` polls container logs for
``Checkpoint created successfully``.

Region: Beam runs hosted customer workloads in the United States unless a
compute pool selects another region
(https://docs.beam.cloud/v2/security/security). ``Pod`` has no region argument.
This module does not pass one, and the reserved pool is not given a ``regions``
list, so SynapsAI Cloud stays on that United States default. Nothing here needs
a region pin to keep deployments in the US.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from provider_service.providers.schema import (
    INSTANCE_STATUS_BUILDING,
    INSTANCE_STATUS_FAILED,
    INSTANCE_STATUS_READY,
    INSTANCE_STATUS_SLEEPING,
    INSTANCE_STATUS_STOPPED,
    ApplicationLogs,
    ModelStatusSummary,
    ProviderInstance,
    assign_warm_by_min_instances,
    status_from_instance_statuses,
)
from provider_service.log import get_logger
from provider_service.settings import (
    DEFAULT_LISTEN_PORT,
    InferenceBackendError,
    beam_api_token,
    beam_checkpoint_enabled,
    cuda_visible_devices_value,
    beam_gpu,
    beam_ondemand_gpu,
    beam_pod_cpu,
    beam_pod_memory,
    beam_registry_secret_names,
    beam_server_unauthenticated,
    beam_target_concurrency,
    function_name_from_app,
    registry_credentials,
    synapsai_app_name,
)

DEFAULT_KEEP_WARM_SECONDS = 300
# Timer starts at container start and ends on the first 200 from the readiness
# path. That 200 is served once the model is loaded. Load can take 30 minutes,
# or 60 for video, and the process still has to import before it binds. Two
# hours covers that. A longer wait only bills a GPU when the image never
# answers the readiness path.
DEFAULT_CHECKPOINT_TIMEOUT = 900 # 15 minutes in seconds should be 7200 in production
_CONTAINER_DEPLOYMENTS: dict[str, str] = {}

logger = get_logger(logger_name="beam")


class BeamBackendError(InferenceBackendError):
    pass


def _require_beam() -> None:
    import beam  # noqa: F401 — configures beta9 settings for Beam Cloud


def _config_context():
    from beta9.abstractions.base import set_channel
    from beta9.config import ConfigContext, settings_context

    token = beam_api_token()
    if token == "":
        raise BeamBackendError(
            "BEAM_TOKEN is required to use the Beam backend. "
            "See https://docs.beam.cloud/v2/reference/py-sdk"
        )
    base = settings_context()
    context = ConfigContext(
        token=token,
        gateway_host=base.gateway_host,
        gateway_port=base.gateway_port,
        api_url=base.api_url,
        auth_url=base.auth_url,
    )
    set_channel(context=context)
    return context


def _gateway():
    from beta9.abstractions.base import get_channel
    from beta9.clients.gateway import GatewayServiceStub

    _require_beam()
    _config_context()
    return GatewayServiceStub(get_channel())


def _upsert_beam_secret(name: str, value: str) -> None:
    """Create or update one Beam workspace secret. Values stay off the logs."""
    from beta9.channel import ServiceClient
    from beta9.clients.secret import (
        CreateSecretRequest,
        GetSecretRequest,
        UpdateSecretRequest,
    )

    context = _config_context()
    client = ServiceClient(config=context)
    try:
        current = client.secret.get_secret(GetSecretRequest(name=name))
        if current.ok and current.secret.value == value:
            return
        if current.ok:
            updated = client.secret.update_secret(UpdateSecretRequest(name=name, value=value))
            if not updated.ok:
                raise BeamBackendError(updated.err_msg or f"Failed to update Beam secret {name}")
            return
        created = client.secret.create_secret(CreateSecretRequest(name=name, value=value))
        if created.ok:
            return
        updated = client.secret.update_secret(UpdateSecretRequest(name=name, value=value))
        if not updated.ok:
            raise BeamBackendError(created.err_msg or f"Failed to create Beam secret {name}")
    finally:
        client.close()


def _ensure_registry_secrets(image: str) -> list[str] | None:
    """Write GHCR credentials into the Beam account and return their secret names."""
    names = beam_registry_secret_names(image)
    if not names:
        return None
    username, password = registry_credentials()
    _upsert_beam_secret(names[0], username)
    _upsert_beam_secret(names[1], password)
    return names


def _inference_image(image: str):
    from beam import Image

    creds = _ensure_registry_secrets(image)
    if creds:
        return Image.from_registry(image, credentials=creds)
    return Image.from_registry(image)


def _autoscaler(
    *,
    min_instances: int,
    max_instances: int,
    scale_up_threshold: float | None,
    tasks_per_container: int | None = None,
):
    from beta9.type import QueueDepthAutoscaler

    minimum = max(0, int(min_instances))
    maximum = max(minimum, int(max_instances), 1)
    if tasks_per_container is None:
        tasks_per_container = beam_target_concurrency(scale_up_threshold)
    return QueueDepthAutoscaler(
        min_containers=minimum,
        max_containers=maximum,
        tasks_per_container=max(1, int(tasks_per_container)),
    )


def _keep_warm_seconds(worker_timeout: float | None) -> int:
    if worker_timeout is None:
        raw = (os.getenv("BEAM_KEEP_WARM_SECONDS") or "").strip()
        if raw == "":
            return DEFAULT_KEEP_WARM_SECONDS
        try:
            return max(0, int(raw))
        except ValueError as exc:
            raise BeamBackendError("BEAM_KEEP_WARM_SECONDS must be an integer") from exc
    return min(3600, max(0, int(float(worker_timeout)) or 0))


def _checkpoint_timeout() -> int:
    raw = (os.getenv("BEAM_CHECKPOINT_TIMEOUT") or "").strip()
    if raw == "":
        return DEFAULT_CHECKPOINT_TIMEOUT
    try:
        value = int(raw)
    except ValueError as exc:
        raise BeamBackendError("BEAM_CHECKPOINT_TIMEOUT must be a positive integer") from exc
    if value <= 0:
        raise BeamBackendError("BEAM_CHECKPOINT_TIMEOUT must be a positive integer")
    return value


def _build_pod(
    *,
    name: str,
    command: list[str],
    image: str,
    gpu: str | list[str],
    gpu_count: int,
    vram_mb: int,
    min_instances: int,
    max_instances: int,
    worker_timeout: float | None,
    scale_up_threshold: float | None,
    hf_token: str | None,
    pool: str | None = None,
    persistent: bool = False,
    tasks_per_container: int | None = None,
    extra_env: dict[str, str] | None = None,
    port: int | None = None,
    readiness_path: str | None = None,
):
    from beam import Pod

    listen = int(port or DEFAULT_LISTEN_PORT)
    env: dict[str, str] = {
        "PORT": str(listen),
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices_value(gpu_count),
    }
    if hf_token:
        env["HF_TOKEN"] = hf_token
    if extra_env:
        env.update({key: value for key, value in extra_env.items() if value})
    pod = Pod(
        name=name,
        entrypoint=list(command),
        ports=[listen],
        image=_inference_image(image),
        gpu=gpu,
        gpu_count=int(gpu_count or 1),
        cpu=beam_pod_cpu(),
        memory=beam_pod_memory(vram_mb, gpu_count),
        env=env,
        keep_warm_seconds=-1 if persistent else _keep_warm_seconds(worker_timeout),
        authorized=not beam_server_unauthenticated(),
        checkpoint_enabled=beam_checkpoint_enabled(),
        checkpoint_readiness_path=str(readiness_path or ""),
        checkpoint_readiness_port=listen,
        checkpoint_readiness_timeout=_checkpoint_timeout(),
        pool=pool,
    )
    pod.autoscaler = _autoscaler(
        min_instances=min_instances,
        max_instances=max_instances,
        scale_up_threshold=scale_up_threshold,
        tasks_per_container=tasks_per_container,
    )
    return pod


def _deployment_url(pod, deployment_id: str) -> str:
    from beta9.clients.gateway import GetUrlRequest

    if not pod.ports:
        raise BeamBackendError("Beam pod is missing exposed ports")
    response = pod.gateway_stub.get_url(
        GetUrlRequest(
            stub_id=pod.stub_id,
            deployment_id=deployment_id,
            url_type="path",
        )
    )
    if not response.ok or not response.url:
        raise BeamBackendError(response.err_msg or "Beam did not return an invocation URL")
    return str(response.url).replace("<PORT>", str(pod.ports[0])).rstrip("/")


def list_deployments() -> list[Any]:
    from beta9.clients.gateway import ListDeploymentsRequest

    gateway = _gateway()
    response = gateway.list_deployments(ListDeploymentsRequest(limit=500))
    if not response.ok:
        raise BeamBackendError(response.err_msg or "Failed to list Beam deployments")
    return list(response.deployments or [])


def list_deployments_named(name: str) -> list[Any]:
    """Every deployment with this exact name, not only the first of 500."""
    from beta9.clients.gateway import ListDeploymentsRequest, StringList

    target = str(name or "")
    if target == "":
        return []
    gateway = _gateway()
    response = gateway.list_deployments(
        ListDeploymentsRequest(filters={"name": StringList([target])}, limit=100)
    )
    if not response.ok:
        raise BeamBackendError(response.err_msg or f"Failed to list Beam deployments named {target}")
    return [
        item
        for item in (response.deployments or [])
        if str(getattr(item, "name", "") or "") == target
    ]


def find_named_deployment(name: str) -> Any | None:
    matches = list_deployments_named(name)
    return matches[0] if matches else None


def find_deployment(model_id: str) -> Any | None:
    from provider_service.providers.beam.capacity import warm_app_name

    warm = find_named_deployment(warm_app_name(model_id))
    if warm is not None:
        return warm
    return find_named_deployment(synapsai_app_name(model_id))


def deploy_inference_pod(
    *,
    model_id: str,
    command: list[str],
    image: str | None = None,
    gpu_count: int = 1,
    vram_mb: int = 24000,
    min_instances: int = 0,
    max_instances: int = 1,
    worker_timeout: float | None = None,
    scale_up_threshold: float | None = None,
    hf_token: str | None = None,
    tasks_per_container: int | None = None,
    extra_env: dict[str, str] | None = None,
    untrusted: bool = False,
    port: int | None = None,
    readiness_path: str | None = None,
) -> dict[str, Any]:
    """Create or update Beam capacity for one model and return its URL.

    Persistent replicas (``min_instances``) are reserved on-demand machines.
    Anything above that minimum is a serverless Pod that scales from zero.
    ``BEAM_ONDEMAND_ONLY=1`` reserves ``max_instances`` machines and skips
    the serverless Pod.
    """
    from provider_service.providers.beam.capacity import (
        ensure_ondemand_pool,
        pool_name_for,
        release_ondemand_pool,
        split_capacity,
        warm_app_name,
    )

    if not command:
        raise BeamBackendError("command is required")
    if not str(image or "").strip():
        raise BeamBackendError("image is required")
    if untrusted:
        from provider_service.providers.beam.guest_gate import (
            APP_PORT,
            LISTEN_PORT,
            edge_token,
            gate_program,
            untrusted_entrypoint,
        )

        secret = edge_token(model_id, beam_api_token())
        if secret == "":
            raise BeamBackendError(
                "BEAM_TOKEN is required to deploy a custom image without placing it in the container"
            )
        command = untrusted_entrypoint(list(command))
        extra_env = {
            **(extra_env or {}),
            "SYNAPS_EDGE_TOKEN": secret,
            "SYNAPS_EDGE_SCRIPT": gate_program(),
            "SYNAPS_LISTEN_PORT": str(LISTEN_PORT),
            "SYNAPS_APP_PORT": str(APP_PORT),
        }
    resolved_image = str(image).strip()
    persistent, burst = split_capacity(min_instances, max_instances)
    if tasks_per_container is None:
        tasks_per_container = beam_target_concurrency(scale_up_threshold)
    tasks_per_container = max(1, int(tasks_per_container))
    pod_gpu = beam_gpu(vram_mb, gpu_count) if burst else ""
    reserved_gpu = beam_ondemand_gpu(vram_mb, gpu_count) if persistent else ""
    burst_name = synapsai_app_name(model_id)
    warm_name = warm_app_name(model_id)
    persistent_url = ""
    burst_url = ""
    persistent_id = ""
    burst_id = ""

    if persistent:
        pool = ensure_ondemand_pool(
            model_id=model_id,
            gpu=reserved_gpu,
            gpu_count=gpu_count,
            nodes=persistent,
        )
        persistent_id, persistent_url = _deploy_named_pod(
            name=warm_name,
            command=command,
            image=resolved_image,
            gpu=reserved_gpu,
            gpu_count=gpu_count,
            vram_mb=vram_mb,
            min_instances=persistent,
            max_instances=persistent,
            worker_timeout=worker_timeout,
            scale_up_threshold=scale_up_threshold,
            hf_token=hf_token,
            pool=pool,
            persistent=True,
            tasks_per_container=tasks_per_container,
            extra_env=extra_env,
            untrusted=untrusted,
            port=port,
            readiness_path=readiness_path,
        )
    else:
        _delete_named_deployment(warm_name)
        release_ondemand_pool(model_id)

    if burst:
        burst_id, burst_url = _deploy_named_pod(
            name=burst_name,
            command=command,
            image=resolved_image,
            gpu=pod_gpu,
            gpu_count=gpu_count,
            vram_mb=vram_mb,
            min_instances=0,
            max_instances=burst,
            worker_timeout=worker_timeout,
            scale_up_threshold=scale_up_threshold,
            hf_token=hf_token,
            tasks_per_container=tasks_per_container,
            extra_env=extra_env,
            untrusted=untrusted,
            port=port,
            readiness_path=readiness_path,
        )
    elif persistent:
        _delete_named_deployment(burst_name)

    url = persistent_url or burst_url
    if not url:
        raise BeamBackendError(f"Beam did not return a URL for {burst_name}")
    return {
        "ok": True,
        "id": persistent_id or burst_id,
        "app_name": warm_name if persistent_url else burst_name,
        "url": url,
        "persistent_url": persistent_url,
        "burst_url": burst_url,
        "persistent_instances": persistent,
        "burst_max": burst,
        "pool": pool_name_for(model_id) if persistent else "",
        "image": resolved_image,
        "gpu": reserved_gpu if persistent_url else pod_gpu,
        "tasks_per_container": tasks_per_container,
        "checkpoint_enabled": beam_checkpoint_enabled(),
    }


def _deploy_named_pod(**kwargs) -> tuple[str, str]:
    name = kwargs["name"]
    kwargs.pop("untrusted", None)
    pod = _build_pod(**kwargs)
    context = _config_context()
    details, ok = pod.deploy(name=name, context=context)
    if not ok:
        raise BeamBackendError(f"Beam failed to deploy pod {name}")
    deployment_id = str(details.get("deployment_id") or "")
    url = str(details.get("invoke_url") or "").rstrip("/")
    if not url and deployment_id:
        url = _deployment_url(pod, deployment_id)
    if not url:
        raise BeamBackendError(f"Beam did not return a URL for {name}")
    if deployment_id:
        _CONTAINER_DEPLOYMENTS[name] = deployment_id
    return deployment_id, url


def _beam_account() -> tuple[str, str, str]:
    """Return ``(api_url, workspace_id, token)`` for the Beam workspace REST API."""
    from beta9.clients.gateway import ExportWorkspaceConfigRequest

    context = _config_context()
    workspace = _gateway().export_workspace_config(ExportWorkspaceConfigRequest())
    workspace_id = str(getattr(workspace, "workspace_id", "") or "")
    token = str(context.token or "")
    if workspace_id == "" or token == "":
        raise BeamBackendError("Beam workspace credentials are required")
    return context.http_url.rstrip("/"), workspace_id, token


def _app_rows(payload: Any) -> tuple[list[Any], str]:
    if isinstance(payload, list):
        return payload, ""
    if not isinstance(payload, dict):
        return [], ""
    rows = payload.get("data")
    if not isinstance(rows, list):
        rows = payload.get("apps") if isinstance(payload.get("apps"), list) else payload.get("items")
    if not isinstance(rows, list):
        rows = []
    cursor = payload.get("next") or payload.get("cursor") or ""
    if not isinstance(cursor, str):
        cursor = ""
    return rows, cursor


def _app_ids_named(name: str) -> list[str]:
    """Dashboard app ids whose name matches, including apps with no deployment left."""
    import httpx

    target = str(name or "")
    if target == "":
        return []
    http_url, workspace_id, token = _beam_account()
    headers = {"Authorization": f"Bearer {token}"}
    found: list[str] = []
    cursor = ""
    for _ in range(5):
        params: dict[str, str] = {"name": target, "limit": "100"}
        if cursor:
            params["cursor"] = cursor
        response = httpx.get(
            f"{http_url}/api/v1/app/{workspace_id}",
            params=params,
            headers=headers,
            timeout=30,
        )
        if response.status_code >= 400:
            raise BeamBackendError(
                f"Failed to list Beam apps (HTTP {response.status_code}): {response.text[:500]}"
            )
        rows, cursor = _app_rows(response.json())
        for row in rows:
            if not isinstance(row, dict) or str(row.get("name") or "") != target:
                continue
            app_id = str(row.get("id") or row.get("external_id") or row.get("app_id") or "")
            if app_id:
                found.append(app_id)
        if cursor == "":
            break
    return found


def _delete_beam_app(app_id: str) -> None:
    """Remove the dashboard app. Deleting a deployment version leaves this behind."""
    import httpx

    target = str(app_id or "").strip()
    if target == "":
        return
    http_url, workspace_id, token = _beam_account()
    response = httpx.delete(
        f"{http_url}/api/v1/app/{workspace_id}/{target}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    if response.status_code in {200, 202, 204, 404}:
        logger.info("Deleted Beam app %s", target)
        return
    raise BeamBackendError(
        f"Failed to delete Beam app {target} (HTTP {response.status_code}): {response.text[:500]}"
    )


def _beam_absence(message: str) -> bool:
    text = message.lower()
    return "not found" in text or "not active" in text or "already" in text


def _remember_delete_error(errors: list[str], label: str, exc: Exception) -> None:
    logger.error("%s: %s", label, exc)
    errors.append(f"{label}: {exc}")


def _stop_named_deployment(gateway, deployment_id: str) -> None:
    """Mark the deployment inactive so Beam force-stops keep-warm containers."""
    from beta9.clients.gateway import StopDeploymentRequest

    response = gateway.stop_deployment(StopDeploymentRequest(id=deployment_id))
    if getattr(response, "ok", False):
        logger.info("Stopped Beam deployment %s", deployment_id)
        return
    message = str(getattr(response, "err_msg", "") or "")
    if _beam_absence(message):
        return
    raise BeamBackendError(message or f"Failed to stop Beam deployment {deployment_id}")


def _stop_deployment_containers(gateway, deployment) -> None:
    """Stop containers while they are still listed. Delete drops them from that list."""
    from beta9.clients.gateway import ListContainersRequest, StopContainerRequest

    deployment_id = str(getattr(deployment, "id", "") or "")
    stub_id = str(getattr(deployment, "stub_id", "") or "")
    response = gateway.list_containers(ListContainersRequest())
    if not getattr(response, "ok", False):
        message = str(getattr(response, "err_msg", "") or "")
        raise BeamBackendError(message or "Failed to list Beam containers")
    for container in response.containers or []:
        container_deployment = str(getattr(container, "deployment_id", "") or "")
        container_stub = str(getattr(container, "stub_id", "") or "")
        matches_deployment = deployment_id != "" and container_deployment == deployment_id
        matches_stub = stub_id != "" and container_stub == stub_id
        if not matches_deployment and not matches_stub:
            continue
        container_id = str(getattr(container, "container_id", "") or "")
        if container_id == "":
            continue
        stopped = gateway.stop_container(StopContainerRequest(container_id=container_id))
        if getattr(stopped, "ok", False):
            logger.info("Stopped Beam container %s", container_id)
            continue
        message = str(getattr(stopped, "err_msg", "") or "")
        if _beam_absence(message):
            continue
        raise BeamBackendError(message or f"Failed to stop Beam container {container_id}")


def _delete_deployment_record(gateway, name: str, deployment_id: str) -> None:
    from beta9.clients.gateway import DeleteDeploymentRequest

    response = gateway.delete_deployment(DeleteDeploymentRequest(id=deployment_id))
    if not getattr(response, "ok", False):
        message = str(getattr(response, "err_msg", "") or "")
        if _beam_absence(message):
            _CONTAINER_DEPLOYMENTS.pop(name, None)
            return
        raise BeamBackendError(message or f"Failed to delete Beam deployment {deployment_id}")
    logger.info("Deleted Beam deployment %s (%s)", name, deployment_id)
    _CONTAINER_DEPLOYMENTS.pop(name, None)


def _delete_named_deployment(name: str) -> None:
    """Stop, delete, and drop the volumes for every revision of this name.

    A failure is recorded and the remaining steps still run. The caller gets
    one error after the deployment, its containers, its volume, and its
    dashboard app have all been attempted.
    """
    from provider_service.providers.beam.capacity import release_app_volumes

    errors: list[str] = []
    matches: list[Any] = []
    try:
        matches = list_deployments_named(name)
    except Exception as exc:
        _remember_delete_error(errors, f"list Beam deployments named {name}", exc)
    app_ids = [
        str(getattr(deployment, "app_id", "") or "")
        for deployment in matches
        if str(getattr(deployment, "app_id", "") or "")
    ]
    try:
        app_ids.extend(_app_ids_named(name))
    except Exception as exc:
        _remember_delete_error(errors, f"list Beam apps named {name}", exc)
    if matches:
        gateway = _gateway()
        for deployment in matches:
            deployment_id = str(getattr(deployment, "id", "") or "")
            if deployment_id == "":
                message = f"Beam deployment {name} has no id and was not deleted"
                logger.error(message)
                errors.append(message)
                continue
            for label, action in (
                (f"stop Beam deployment {name} ({deployment_id})", lambda deployment_id=deployment_id: _stop_named_deployment(gateway, deployment_id)),
                (f"stop Beam containers for {name} ({deployment_id})", lambda deployment=deployment: _stop_deployment_containers(gateway, deployment)),
                (f"delete Beam deployment {name} ({deployment_id})", lambda deployment_id=deployment_id: _delete_deployment_record(gateway, name, deployment_id)),
            ):
                try:
                    action()
                except Exception as exc:
                    _remember_delete_error(errors, label, exc)
    try:
        release_app_volumes(name)
    except Exception as exc:
        _remember_delete_error(errors, f"delete Beam volumes for {name}", exc)
    seen: set[str] = set()
    for app_id in app_ids:
        if app_id in seen:
            continue
        seen.add(app_id)
        try:
            _delete_beam_app(app_id)
        except Exception as exc:
            _remember_delete_error(errors, f"delete Beam app {app_id}", exc)
        _CONTAINER_DEPLOYMENTS.pop(name, None)
    if errors:
        raise BeamBackendError("; ".join(errors))


def lookup_pod_url(model_id: str) -> str:
    deployment = find_deployment(model_id)
    if deployment is None:
        raise BeamBackendError(f"Beam deployment {synapsai_app_name(model_id)} is not deployed")
    return _url_of_deployment(deployment)


def describe_routes(model_id: str) -> dict[str, Any]:
    """URLs for the warm on-demand Pod and the serverless burst Pod."""
    from provider_service.providers.beam.capacity import (
        _pool_by_name,
        _ready_machines,
        pool_name_for,
        warm_app_name,
    )

    warm = find_named_deployment(warm_app_name(model_id))
    burst = find_named_deployment(synapsai_app_name(model_id))
    persistent_url = _url_of_deployment(warm) if warm is not None else ""
    burst_url = _url_of_deployment(burst) if burst is not None else ""
    persistent = 0
    if persistent_url:
        pool = _pool_by_name(_gateway(), pool_name_for(model_id))
        persistent = max(1, _ready_machines(pool))
    return {
        "persistent_url": persistent_url,
        "burst_url": burst_url,
        "persistent_instances": persistent,
    }


def _url_of_deployment(deployment) -> str:
    deployment_id = str(getattr(deployment, "id", "") or "")
    stub_id = str(getattr(deployment, "stub_id", "") or "")
    from beam import Pod

    _require_beam()
    _config_context()
    pod = Pod(name=str(getattr(deployment, "name", "") or ""), ports=[DEFAULT_LISTEN_PORT])
    pod.stub_id = stub_id
    pod.deployment_id = deployment_id
    return _deployment_url(pod, deployment_id)


def stop_inference_pod(model_id: str) -> dict[str, Any]:
    from provider_service.providers.beam.capacity import (
        release_ondemand_pool,
        warm_app_name,
    )

    name = synapsai_app_name(model_id)
    warm = warm_app_name(model_id)
    errors: list[str] = []
    for label, action in (
        (f"burst deployment {name}", lambda: _delete_named_deployment(name)),
        (f"warm deployment {warm}", lambda: _delete_named_deployment(warm)),
        (f"on-demand pool for {model_id}", lambda: release_ondemand_pool(model_id)),
    ):
        try:
            action()
        except Exception as exc:
            logger.error("Beam teardown failed for %s: %s", label, exc)
            errors.append(f"{label}: {exc}")
    if errors:
        raise BeamBackendError("; ".join(errors))
    return {"ok": True, "app_name": name}


def update_pod_autoscaler(
    model_id: str,
    *,
    min_instances: int,
    max_instances: int,
    worker_timeout: float | None = None,
    scale_up_threshold: float | None = None,
) -> dict[str, Any]:
    """Scaling changes are applied by redeploying the model, not by this call.

    Reserving or releasing machines here would race the redeploy that follows
    an update.
    """
    del worker_timeout, scale_up_threshold
    from provider_service.providers.beam.capacity import split_capacity

    persistent, burst = split_capacity(min_instances, max_instances)
    return {
        "ok": True,
        "app_name": synapsai_app_name(model_id),
        "persistent_instances": persistent,
        "burst_max": burst,
        "applied_on_redeploy": True,
    }


def _container_state(status: str) -> str:
    normalized = str(status or "").strip().upper()
    if normalized in {"RUNNING", "READY", "WORKING", "IDLE"}:
        return "Running"
    if normalized in {"PENDING", "SCHEDULED", "RESTORING", "RESUMING", "CREATING", "STARTING"}:
        return "Restoring"
    if normalized in {"STOPPING", "TERMINATING", "TERMINATED", "STOPPED", "DELETED"}:
        return "Stopping"
    if normalized in {"FAILED", "ERROR"}:
        return "Failed"
    return normalized or "Restoring"


def _container_status(status: str) -> str:
    normalized = str(status or "").strip().upper()
    if normalized in {"RUNNING", "READY", "WORKING", "IDLE"}:
        return INSTANCE_STATUS_READY
    if normalized in {
        "PENDING",
        "SCHEDULED",
        "RESTORING",
        "RESUMING",
        "CREATING",
        "STARTING",
        "SLEEPING",
        "STANDBY",
    }:
        return INSTANCE_STATUS_SLEEPING
    if normalized in {"STOPPING", "TERMINATING", "TERMINATED", "STOPPED", "DELETED", ""}:
        return INSTANCE_STATUS_STOPPED
    if normalized in {"FAILED", "ERROR"}:
        return INSTANCE_STATUS_FAILED
    if normalized == "BUILDING":
        return INSTANCE_STATUS_BUILDING
    return INSTANCE_STATUS_SLEEPING


def normalize_pod_instance(pod: dict[str, Any]) -> ProviderInstance:
    from provider_service.providers.beam.capacity import function_name_for_app, is_warm_app

    app_name = str(pod.get("app_name") or "").strip()
    funcname = str(pod.get("funcname") or "").strip()
    if not funcname and app_name:
        funcname = function_name_for_app(app_name, function_name_from_app)
    name = str(pod.get("name") or pod.get("id") or funcname)
    pod_id = str(pod.get("id") or name or funcname)
    raw_state = str(pod.get("state") or "").strip()
    raw_status = str(pod.get("status") or raw_state).strip()
    state = raw_state or _container_state(raw_status)
    status = _container_status(raw_status)
    if isinstance(pod.get("warm"), bool):
        warm = pod["warm"]
    else:
        warm = is_warm_app(app_name or name)
    return ProviderInstance(
        id=pod_id,
        name=name,
        funcname=funcname,
        state=state,
        status=status,
        warm=warm,
        backend="beam",
    )


def summarize_pod_instances(
    pods: list[dict[str, Any]],
    *,
    min_instances: int = 0,
) -> ModelStatusSummary:
    from provider_service.providers.beam.capacity import is_warm_app

    instances = [normalize_pod_instance(pod) for pod in pods]
    has_explicit_warm_info = any(
        isinstance(pod.get("warm"), bool)
        or bool(str(pod.get("app_name") or "").strip())
        or is_warm_app(str(pod.get("name") or ""))
        for pod in pods
    )
    if not has_explicit_warm_info and min_instances > 0:
        instances = assign_warm_by_min_instances(instances, min_instances)
    status, active = status_from_instance_statuses([inst.status for inst in instances])
    serving = [inst for inst in instances if inst.status == INSTANCE_STATUS_READY]
    return ModelStatusSummary(
        status=status,
        active_instances=active,
        serving_instances=serving,
    )


def list_pod_instances() -> list[dict[str, Any]]:
    """Active Beam containers for SynapsAI pods in the standard provider format."""
    from beta9.clients.gateway import ListContainersRequest
    from provider_service.providers.beam.capacity import (
        function_name_for_app,
        is_warm_app,
    )

    gateway = _gateway()
    response = gateway.list_containers(ListContainersRequest())
    if not response.ok:
        return []
    deployments = {
        str(getattr(item, "id", "") or ""): item
        for item in list_deployments()
        if str(getattr(item, "name", "") or "").startswith("synapsai-")
    }
    deployment_by_stub = {
        str(getattr(item, "stub_id", "") or ""): item for item in deployments.values()
    }
    rows: list[dict[str, Any]] = []
    for container in response.containers or []:
        deployment_id = str(getattr(container, "deployment_id", "") or "")
        stub_id = str(getattr(container, "stub_id", "") or "")
        deployment = deployments.get(deployment_id) or deployment_by_stub.get(stub_id)
        if deployment is None:
            continue
        name = str(getattr(deployment, "name", "") or "")
        if not name.startswith("synapsai-"):
            continue
        container_id = str(getattr(container, "container_id", "") or "")
        if container_id == "":
            continue
        raw_status = str(getattr(container, "status", "") or "")
        rows.append(
            {
                "id": container_id,
                "name": container_id,
                "funcname": function_name_for_app(name, function_name_from_app),
                "state": _container_state(raw_status),
                "status": _container_status(raw_status),
                "warm": is_warm_app(name),
                "backend": "beam",
                "app_name": name,
                "deployment_id": deployment_id,
            }
        )
    return rows


def list_pod_functions() -> list[dict[str, str]]:
    rows = []
    for deployment in list_deployments():
        name = str(getattr(deployment, "name", "") or "")
        if not name.startswith("synapsai-"):
            continue
        from provider_service.providers.beam.capacity import (
            function_name_for_app,
        )

        rows.append(
            {
                "name": function_name_for_app(name, function_name_from_app),
                "app_name": name,
                "id": str(getattr(deployment, "id", "") or ""),
            }
        )
    return rows


def _beam_log_text(records: Any, *, tail: int) -> str:
    if not isinstance(records, list):
        return ""
    messages: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        message = record.get("message")
        if message is None:
            continue
        messages.append(str(message))
    if tail > 0 and len(messages) > tail:
        messages = messages[-tail:]
    return "\n".join(messages)


def _read_beam_log_records(
    *,
    object_id: str,
    object_type: str,
    tail: int,
    workspace_id: str,
    http_url: str,
    token: str,
    start_seq: int | None = None,
) -> list[dict[str, Any]]:
    import httpx

    limit = max(1, min(int(tail), 1000))
    params: dict[str, Any] = {
        "object_id": object_id,
        "object_type": object_type,
        "limit": limit,
    }
    if start_seq is not None:
        params["seq_num"] = start_seq
    response = httpx.get(
        f"{http_url.rstrip('/')}/api/v1/logs/{workspace_id}",
        params=params,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if response.status_code >= 400:
        raise BeamBackendError(
            f"Beam log request failed (HTTP {response.status_code}): {response.text[:500]}"
        )
    payload = response.json()
    logs = payload.get("logs") if isinstance(payload, dict) else None
    if not isinstance(logs, list):
        return []
    return [record for record in logs if isinstance(record, dict)]


def _read_beam_object_logs(
    *,
    object_id: str,
    object_type: str,
    tail: int,
    workspace_id: str,
    http_url: str,
    token: str,
) -> str:
    records = _read_beam_log_records(
        object_id=object_id,
        object_type=object_type,
        tail=tail,
        workspace_id=workspace_id,
        http_url=http_url,
        token=token,
    )
    return _beam_log_text(records, tail=max(1, min(int(tail), 1000)))


def _log_record_source(record: dict[str, Any]) -> str:
    """Beam's log API calls this field ``stream``. Some payloads use ``source``."""
    source = record.get("source")
    if source is None or str(source).strip() == "":
        source = record.get("stream")
    return str(source or "").strip().lower()


def _log_record_seq(record: dict[str, Any]) -> int | None:
    raw = record.get("seq_num")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _decode_application_cursor(cursor: str) -> dict[str, dict[str, Any]]:
    if not str(cursor or "").strip():
        return {}
    try:
        payload = json.loads(cursor)
    except json.JSONDecodeError:
        return {}
    containers = payload.get("containers") if isinstance(payload, dict) else None
    if not isinstance(containers, dict):
        return {}
    decoded: dict[str, dict[str, Any]] = {}
    for container_id, state in containers.items():
        if not isinstance(state, dict):
            continue
        try:
            seq = int(state.get("seq") or 0)
        except (TypeError, ValueError):
            seq = 0
        seen = [str(item) for item in (state.get("seen") or []) if str(item)]
        decoded[str(container_id)] = {"seq": max(0, seq), "seen": seen[-300:]}
    return decoded


def _encode_application_cursor(containers: dict[str, dict[str, Any]]) -> str:
    if not containers:
        return ""
    return json.dumps({"containers": containers}, sort_keys=True, separators=(",", ":"))


def _stdout_lines_from_records(
    records: list[dict[str, Any]],
    *,
    last_seq: int,
    seen: set[str],
) -> tuple[list[str], int, list[str]]:
    ordered = sorted(records, key=lambda record: _log_record_seq(record) or 0)
    max_seq = last_seq
    lines: list[str] = []
    seen_order = [item for item in seen]
    seen_set = set(seen_order)
    for record in ordered:
        seq = _log_record_seq(record)
        if seq is not None:
            max_seq = max(max_seq, seq)
            if seq <= last_seq:
                continue
        if _log_record_source(record) != "stdout":
            continue
        message = str(record.get("message") or "").strip()
        if message == "":
            continue
        if seq is None:
            fingerprint = f"{record.get('timestamp')}|{message}"
            if fingerprint in seen_set:
                continue
            seen_set.add(fingerprint)
            seen_order.append(fingerprint)
        lines.append(message)
    return lines, max_seq, seen_order[-300:]


_CHECKPOINT_CREATED = "Checkpoint created successfully"
_CHECKPOINT_COPY_FAILED = "Failed to copy checkpoint filesystem state"
# Beam keeps archiving and uploading after the readiness check. The readiness
# timeout does not cover that copy.
_CHECKPOINT_ARCHIVE_GRACE_S = 15 * 60


def checkpoint_status_from_logs(text: str) -> str:
    if _CHECKPOINT_CREATED.lower() in text.lower():
        return "created"
    if _CHECKPOINT_COPY_FAILED.lower() in text.lower():
        return "failed"
    return "pending"


def wait_for_checkpoint(model_id: str, *, timeout_s: float | None = None, poll_s: float = 5.0) -> None:
    """Block until Beam has finished the container snapshot for ``model_id``.

    The pod URL is reachable as soon as the readiness path returns 200. Inference
    during the following archive writes Triton files under ``/tmp`` and makes
    ``tar`` fail with "file changed as we read it".
    """
    if not beam_checkpoint_enabled():
        return
    limit = _checkpoint_timeout() + _CHECKPOINT_ARCHIVE_GRACE_S if timeout_s is None else timeout_s
    deadline = time.monotonic() + max(0.0, float(limit))
    last_logged = 0.0
    while True:
        status = checkpoint_status_from_logs(get_model_logs(model_id, tail=1000))
        if status == "created":
            logger.info("Beam checkpoint created for %s", model_id)
            return
        if status == "failed":
            raise BeamBackendError(
                "Beam failed to copy the checkpoint filesystem while the container was still writing files"
            )
        now = time.monotonic()
        if now >= deadline:
            raise BeamBackendError(f"Timed out waiting for the Beam checkpoint for {model_id}")
        if now - last_logged >= 30.0:
            logger.info("Waiting for Beam checkpoint for %s", model_id)
            last_logged = now
        time.sleep(min(poll_s, max(0.0, deadline - now)))


def _model_log_targets(model_id: str, *, include_deployments: bool) -> list[tuple[str, str, str]]:
    from provider_service.providers.beam.capacity import warm_app_name
    from provider_service.settings import function_name_from_app, synapsai_app_name

    wanted = function_name_from_app(synapsai_app_name(model_id))
    targets: list[tuple[str, str, str]] = []
    for row in list_pod_instances():
        if str(row.get("funcname") or "") != wanted:
            continue
        container_id = str(row.get("id") or "")
        if container_id == "":
            continue
        state = str(row.get("state") or "container")
        targets.append((container_id, "BETA9_CONTAINER", f"{container_id} ({state})"))
    if targets or not include_deployments:
        return targets
    for name in (synapsai_app_name(model_id), warm_app_name(model_id)):
        deployment = find_named_deployment(name)
        if deployment is None:
            continue
        deployment_id = str(getattr(deployment, "id", "") or "")
        if deployment_id == "":
            continue
        targets.append((deployment_id, "BETA9_DEPLOYMENT", name))
    return targets


def _beam_log_auth() -> tuple[str, str, str]:
    from beta9.clients.gateway import ExportWorkspaceConfigRequest

    context = _config_context()
    workspace = _gateway().export_workspace_config(ExportWorkspaceConfigRequest())
    workspace_id = str(getattr(workspace, "workspace_id", "") or "")
    if workspace_id == "":
        raise BeamBackendError("Beam did not return a workspace id for log lookup")
    token = str(context.token or "")
    if token == "":
        raise BeamBackendError("BEAM_TOKEN is required to read Beam logs")
    return workspace_id, str(context.http_url or ""), token


def get_model_logs(model_id: str, *, tail: int = 200) -> str:
    """Recent container output for this model's Beam pods.

    Running containers are read first. When none are listed, the deployment
    log is used so a replica that already exited can still be inspected.
    """
    targets = _model_log_targets(model_id, include_deployments=True)
    if not targets:
        return ""

    workspace_id, http_url, token = _beam_log_auth()
    sections: list[str] = []
    for object_id, object_type, title in targets:
        body = _read_beam_object_logs(
            object_id=object_id,
            object_type=object_type,
            tail=tail,
            workspace_id=workspace_id,
            http_url=http_url,
            token=token,
        )
        sections.append(f"----- {title} -----\n{body}")
    return "\n".join(sections)


def read_application_logs(model_id: str, *, cursor: str = "") -> ApplicationLogs:
    """Stdout from this model's running containers, excluding Beam system logs.

    ``cursor`` remembers the last sequence number seen for each container so a
    later read does not return the same lines.
    """
    targets = _model_log_targets(model_id, include_deployments=False)
    previous = _decode_application_cursor(cursor)
    if not targets:
        return ApplicationLogs(lines=(), cursor=cursor)

    workspace_id, http_url, token = _beam_log_auth()
    lines: list[str] = []
    containers: dict[str, dict[str, Any]] = {}
    for object_id, object_type, _title in targets:
        state = previous.get(object_id) or {}
        last_seq = int(state.get("seq") or 0)
        seen = {str(item) for item in (state.get("seen") or [])}
        records = _read_beam_log_records(
            object_id=object_id,
            object_type=object_type,
            tail=1000,
            workspace_id=workspace_id,
            http_url=http_url,
            token=token,
            start_seq=last_seq + 1 if last_seq > 0 else None,
        )
        new_lines, max_seq, seen_order = _stdout_lines_from_records(
            records,
            last_seq=last_seq,
            seen=seen,
        )
        lines.extend(new_lines)
        saved: dict[str, Any] = {"seq": max_seq}
        if seen_order:
            saved["seen"] = seen_order
        containers[object_id] = saved
    return ApplicationLogs(lines=tuple(lines), cursor=_encode_application_cursor(containers))


def stop_pod_container(container_id: str) -> dict[str, Any]:
    from beta9.clients.gateway import StopContainerRequest

    target = str(container_id or "").strip()
    if target == "":
        raise BeamBackendError("container id is required")
    gateway = _gateway()
    response = gateway.stop_container(StopContainerRequest(container_id=target))
    if not response.ok:
        message = str(getattr(response, "err_msg", "") or "").lower()
        if "not found" in message:
            return {"ok": True, "missing": True, "id": target}
        raise BeamBackendError(
            getattr(response, "err_msg", None) or f"Failed to stop Beam container {target}"
        )
    return {"ok": True, "id": target}
