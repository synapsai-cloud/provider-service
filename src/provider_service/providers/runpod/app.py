"""Deploy a caller-supplied image as a Runpod load-balancing endpoint.

Runpod docs used here:

- Create: https://docs.runpod.io/api-reference-v2/serverless/create-a-serverless-endpoint
- Workers: https://docs.runpod.io/api-reference-v2/serverless/list-serverless-endpoint-workers
- Worker logs: https://docs.runpod.io/api-reference-v2/serverless/stream-serverless-worker-logs
- Update: https://docs.runpod.io/api-reference-v2/serverless/update-a-serverless-endpoint
- Delete: https://docs.runpod.io/api-reference-v2/serverless/delete-a-serverless-endpoint
- Load balancer: https://docs.runpod.io/serverless/load-balancing/overview
- GPU pools: https://docs.runpod.io/references/gpu-types

A load-balancing endpoint sends HTTP straight to the container port the caller names.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

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
    InferenceBackendError,
    cuda_visible_devices_value,
    function_name_from_app,
    registry_credentials,
    runpod_api_base,
    runpod_api_key,
    runpod_gpu_pool,
    runpod_target_concurrency,
    synapsai_app_name,
)

DEFAULT_DISK_GB = 80
DEFAULT_EXECUTION_TIMEOUT_MS = 1_800_000
_REGISTRY_NAME = "synapsai-ghcr"
_WORKER_ENDPOINTS: dict[str, str] = {}


class RunpodBackendError(InferenceBackendError):
    pass


def _api_key() -> str:
    key = runpod_api_key()
    if key == "":
        raise RunpodBackendError("RUNPOD_API_KEY is required to use the Runpod backend")
    return key


def _positive_int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RunpodBackendError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RunpodBackendError(f"{name} must be a positive integer")
    return value


def _disk_gb() -> int:
    return _positive_int_env("RUNPOD_DISK_GB", DEFAULT_DISK_GB)


def _execution_timeout_ms(worker_timeout: float | None) -> int:
    if worker_timeout is None:
        return _positive_int_env("RUNPOD_EXECUTION_TIMEOUT_MS", DEFAULT_EXECUTION_TIMEOUT_MS)
    seconds = max(1, int(float(worker_timeout)))
    return max(seconds * 1000, 60_000)


def _idle_timeout(worker_timeout: float | None) -> int:
    if worker_timeout is None:
        return 300
    return min(3600, max(1, int(float(worker_timeout)) or 1))


def _request(method: str, path: str, *, json_body: dict | None = None, params: dict | None = None) -> Any:
    url = f"{runpod_api_base()}{path}"
    try:
        response = httpx.request(
            method,
            url,
            headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
            json=json_body,
            params=params,
            timeout=120,
        )
    except httpx.HTTPError as exc:
        raise RunpodBackendError(f"Runpod request failed: {exc}") from exc
    if response.status_code == 204 or response.content == b"":
        return None
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}
    if response.status_code >= 400:
        detail = ""
        if isinstance(payload, dict):
            detail = str(payload.get("detail") or payload.get("title") or payload)
        raise RunpodBackendError(f"Runpod {method} {path} failed ({response.status_code}): {detail}")
    return payload


def _error_is_missing(exc: Exception) -> bool:
    text = str(exc).lower()
    return "404" in text or "not found" in text


def list_endpoints() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params = {"cursor": cursor} if cursor else None
        payload = _request("GET", "/v2/serverless", params=params) or {}
        if isinstance(payload, list):
            rows.extend(item for item in payload if isinstance(item, dict))
            break
        rows.extend(item for item in (payload.get("endpoints") or []) if isinstance(item, dict))
        page = payload.get("pagination") or {}
        if not page.get("hasNextPage"):
            break
        cursor = page.get("nextCursor")
        if not cursor:
            break
    return rows


def endpoint_storage_gb(endpoint: dict[str, Any] | None) -> float:
    """Disk GB Runpod recorded on this endpoint. Missing endpoints are 0."""
    if not isinstance(endpoint, dict):
        return 0.0
    template = endpoint.get("template") if isinstance(endpoint.get("template"), dict) else {}
    raw = endpoint.get("disk")
    if raw is None:
        raw = template.get("disk")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def find_endpoint(model_id: str) -> dict[str, Any] | None:
    name = synapsai_app_name(model_id)
    for endpoint in list_endpoints():
        if str(endpoint.get("name") or "") == name:
            return endpoint
    return None


def _endpoint_url(endpoint: dict[str, Any]) -> str:
    urls = endpoint.get("requestUrls") or {}
    base = str(urls.get("base") or "").rstrip("/")
    if base:
        return base
    endpoint_id = str(endpoint.get("id") or "").strip()
    if endpoint_id == "":
        raise RunpodBackendError("Runpod endpoint did not include an id")
    return f"https://{endpoint_id}.api.runpod.ai"


def _worker_bounds(min_instances: int, max_instances: int) -> tuple[int, int]:
    minimum = max(0, int(min_instances))
    maximum = max(minimum, int(max_instances), 1)
    return minimum, maximum


def _scaling_body(
    scale_up_threshold: float | None,
    request_concurrency: int | None = None,
) -> dict[str, Any]:
    count = (
        runpod_target_concurrency(scale_up_threshold)
        if request_concurrency is None
        else max(1, int(request_concurrency))
    )
    return {
        "type": "REQUEST_COUNT",
        "requestCount": count,
    }


def _ensure_registry(image: str) -> str | None:
    if not str(image).startswith("ghcr.io/"):
        return None
    username, password = registry_credentials()
    if username == "" or password == "":
        raise RunpodBackendError(
            "Pulling a private image on Runpod requires GHCR_INFERENCE_IMAGE_USERNAME and GHCR_INFERENCE_IMAGE_READ_TOKEN"
        )
    payload = _request("GET", "/v2/registries") or {}
    registries = payload.get("registries") if isinstance(payload, dict) else payload
    if isinstance(registries, list):
        for registry in registries:
            if not isinstance(registry, dict):
                continue
            if str(registry.get("name") or "") == _REGISTRY_NAME and registry.get("id"):
                return str(registry["id"])
    created = _request(
        "POST",
        "/v2/registries",
        json_body={"name": _REGISTRY_NAME, "username": username, "password": password},
    ) or {}
    registry_id = str(created.get("id") or "")
    if registry_id == "":
        raise RunpodBackendError("Runpod did not return a registry credential id")
    return registry_id


def _container_body(
    *,
    command: list[str],
    image: str,
    gpu_count: int,
    vram_mb: int,
    min_instances: int,
    max_instances: int,
    worker_timeout: float | None,
    scale_up_threshold: float | None,
    hf_token: str | None,
    request_concurrency: int | None = None,
    extra_env: dict[str, str] | None = None,
    port: int | None = None,
    health_path: str | None = None,
) -> dict[str, Any]:
    pool, count = runpod_gpu_pool(vram_mb, gpu_count)
    minimum, maximum = _worker_bounds(min_instances, max_instances)
    listen = int(port or DEFAULT_LISTEN_PORT)
    env = {
        "PORT": str(listen),
        "PORT_HEALTH": str(listen),
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices_value(gpu_count),
    }
    if health_path:
        env["HEALTH_CHECK_PATH"] = health_path
    if hf_token:
        env["HF_TOKEN"] = hf_token
    if extra_env:
        env.update({key: value for key, value in extra_env.items() if value})
    body: dict[str, Any] = {
        "image": image,
        "type": "LOAD_BALANCER",
        "entrypoint": list(command),
        "ports": [f"{listen}/http"],
        "disk": _disk_gb(),
        "env": env,
        "gpu": {"pools": [pool], "count": count},
        "workers": {
            "min": minimum,
            "max": maximum,
            "idleTimeout": _idle_timeout(worker_timeout),
        },
        "scaling": _scaling_body(scale_up_threshold, request_concurrency),
        "timeout": _execution_timeout_ms(worker_timeout),
    }
    registry_id = _ensure_registry(image)
    if registry_id:
        body["registry"] = registry_id
    return body


def deploy_inference_endpoint(
    *,
    model_id: str,
    command: list[str],
    image: str,
    gpu_count: int = 1,
    vram_mb: int = 24000,
    min_instances: int = 0,
    max_instances: int = 1,
    worker_timeout: float | None = None,
    scale_up_threshold: float | None = None,
    hf_token: str | None = None,
    request_concurrency: int | None = None,
    extra_env: dict[str, str] | None = None,
    port: int | None = None,
    health_path: str | None = None,
) -> dict[str, Any]:
    """Create or replace the Runpod endpoint for one model and return its URL."""
    if not str(image or "").strip():
        raise RunpodBackendError("image is required")
    resolved_image = str(image).strip()
    if not command:
        raise RunpodBackendError("command is required")
    name = synapsai_app_name(model_id)
    body = _container_body(
        command=command,
        image=resolved_image,
        gpu_count=gpu_count,
        vram_mb=vram_mb,
        min_instances=min_instances,
        max_instances=max_instances,
        worker_timeout=worker_timeout,
        scale_up_threshold=scale_up_threshold,
        hf_token=hf_token,
        request_concurrency=request_concurrency,
        extra_env=extra_env,
        port=port,
        health_path=health_path,
    )
    existing = find_endpoint(model_id)
    if existing is not None:
        endpoint_id = str(existing.get("id") or "")
        update = dict(body)
        update.pop("type", None)
        created = _request("PATCH", f"/v2/serverless/{endpoint_id}", json_body=update) or existing
    else:
        body["name"] = name
        created = _request("POST", "/v2/serverless", json_body=body) or {}
    if not isinstance(created, dict):
        raise RunpodBackendError(f"Runpod did not return an endpoint for {name}")
    url = _endpoint_url(created)
    return {
        "ok": True,
        "id": created.get("id"),
        "app_name": name,
        "url": url,
        "image": resolved_image,
        "gpu": body["gpu"]["pools"][0],
    }


def lookup_endpoint_url(model_id: str) -> str:
    endpoint = find_endpoint(model_id)
    if endpoint is None:
        raise RunpodBackendError(f"Runpod endpoint {synapsai_app_name(model_id)} is not deployed")
    return _endpoint_url(endpoint)


def stop_inference_endpoint(model_id: str) -> dict[str, Any]:
    name = synapsai_app_name(model_id)
    endpoint = find_endpoint(model_id)
    if endpoint is None:
        return {"ok": True, "missing": True, "app_name": name}
    endpoint_id = str(endpoint.get("id") or "")
    try:
        _request("DELETE", f"/v2/serverless/{endpoint_id}")
    except RunpodBackendError as exc:
        if _error_is_missing(exc):
            return {"ok": True, "missing": True, "app_name": name}
        raise
    return {"ok": True, "app_name": name, "id": endpoint_id}


def update_endpoint_autoscaler(
    model_id: str,
    *,
    min_instances: int,
    max_instances: int,
    worker_timeout: float | None = None,
    scale_up_threshold: float | None = None,
) -> dict[str, Any]:
    endpoint = find_endpoint(model_id)
    if endpoint is None:
        raise RunpodBackendError(f"Runpod endpoint {synapsai_app_name(model_id)} is not deployed")
    minimum, maximum = _worker_bounds(min_instances, max_instances)
    workers: dict[str, Any] = {"min": minimum, "max": maximum}
    if worker_timeout is not None:
        workers["idleTimeout"] = _idle_timeout(worker_timeout)
    endpoint_id = str(endpoint.get("id") or "")
    _request(
        "PATCH",
        f"/v2/serverless/{endpoint_id}",
        json_body={"workers": workers, "scaling": _scaling_body(scale_up_threshold)},
    )
    return {"ok": True, "id": endpoint_id, "app_name": synapsai_app_name(model_id)}


def _worker_state(status: str) -> str:
    normalized = str(status or "").strip().upper()
    if normalized in {"RUNNING", "READY", "WORKING"}:
        return "Running"
    if normalized == "IDLE":
        return "Idle"
    if normalized in {"UNHEALTHY", "FAILED", "ERROR"}:
        return "Failed"
    if normalized in {"TERMINATING", "TERMINATED", "STOPPING", "STOPPED", "COMPLETED", "DELETED"}:
        return "Stopping"
    if normalized in {"INITIALIZING", "THROTTLED", "PENDING", "SCHEDULED", "RESTORING", "RESUMING", "CREATING", "STARTING"}:
        return "Restoring"
    return normalized or "Restoring"


def _worker_status(status: str) -> str:
    normalized = str(status or "").strip().upper()
    if normalized in {"RUNNING", "IDLE", "READY", "WORKING"}:
        return INSTANCE_STATUS_READY
    if normalized in {"UNHEALTHY", "FAILED", "ERROR"}:
        return INSTANCE_STATUS_FAILED
    if normalized in {"TERMINATING", "TERMINATED", "STOPPING", "STOPPED", "COMPLETED", "DELETED", ""}:
        return INSTANCE_STATUS_STOPPED
    if normalized == "BUILDING":
        return INSTANCE_STATUS_BUILDING
    return INSTANCE_STATUS_SLEEPING


def normalize_endpoint_instance(pod: dict[str, Any]) -> ProviderInstance:
    app_name = str(pod.get("app_name") or "").strip()
    funcname = str(pod.get("funcname") or "").strip()
    if not funcname and app_name:
        funcname = function_name_from_app(app_name)
    name = str(pod.get("name") or pod.get("id") or funcname)
    pod_id = str(pod.get("id") or name or funcname)
    raw_state = str(pod.get("state") or "").strip()
    raw_status = str(pod.get("status") or raw_state).strip()
    state = raw_state or _worker_state(raw_status)
    status = _worker_status(raw_status)
    return ProviderInstance(
        id=pod_id,
        name=name,
        funcname=funcname,
        state=state,
        status=status,
        warm=bool(pod.get("warm", False)),
        backend="runpod",
    )


def summarize_endpoint_instances(
    pods: list[dict[str, Any]],
    *,
    min_instances: int = 0,
) -> ModelStatusSummary:
    instances = [normalize_endpoint_instance(pod) for pod in pods]
    instances = assign_warm_by_min_instances(instances, min_instances)
    status, active = status_from_instance_statuses([inst.status for inst in instances])
    serving = [inst for inst in instances if inst.status == INSTANCE_STATUS_READY]
    return ModelStatusSummary(
        status=status,
        active_instances=active,
        serving_instances=serving,
    )


def list_endpoint_instances() -> list[dict[str, Any]]:
    """Active Runpod workers for SynapsAI endpoints in the standard provider format."""
    rows: list[dict[str, Any]] = []
    for endpoint in list_endpoints():
        name = str(endpoint.get("name") or "")
        if not name.startswith("synapsai-"):
            continue
        endpoint_id = str(endpoint.get("id") or "")
        funcname = function_name_from_app(name)
        minimum = max(
            0,
            int(((endpoint.get("workers") or {}).get("min") or endpoint.get("workersMin") or 0)),
        )
        workers = _list_workers(endpoint_id)
        if not workers:
            continue
        valid_workers = [
            worker
            for worker in workers
            if str(worker.get("id") or "") != ""
        ]
        valid_workers.sort(
            key=lambda worker: (
                0 if _worker_status(str(worker.get("status") or "")) == INSTANCE_STATUS_READY else 1,
                str(worker.get("id") or ""),
            )
        )
        warm_remaining = minimum
        for worker in valid_workers:
            worker_id = str(worker.get("id") or "")
            _WORKER_ENDPOINTS[worker_id] = endpoint_id
            raw_status = str(worker.get("status") or "")
            status = _worker_status(raw_status)
            is_warm = False
            if status == INSTANCE_STATUS_READY and warm_remaining > 0:
                warm_remaining -= 1
                is_warm = True
            rows.append(
                {
                    "id": worker_id,
                    "name": worker_id,
                    "funcname": funcname,
                    "state": _worker_state(raw_status),
                    "status": status,
                    "warm": is_warm,
                    "backend": "runpod",
                    "app_name": name,
                    "endpoint_id": endpoint_id,
                }
            )
    return rows


def _list_workers(endpoint_id: str) -> list[dict[str, Any]]:
    if endpoint_id == "":
        return []
    try:
        payload = _request("GET", f"/v2/serverless/{endpoint_id}/workers") or {}
    except RunpodBackendError:
        return []
    workers = payload.get("workers") if isinstance(payload, dict) else None
    if not isinstance(workers, list):
        return []
    return [worker for worker in workers if isinstance(worker, dict)]


def list_endpoint_functions() -> list[dict[str, str]]:
    rows = []
    for endpoint in list_endpoints():
        name = str(endpoint.get("name") or "")
        if not name.startswith("synapsai-"):
            continue
        rows.append(
            {
                "name": function_name_from_app(name),
                "app_name": name,
                "id": str(endpoint.get("id") or ""),
            }
        )
    return rows


def _sse_data_line(raw: str) -> str | None:
    text = str(raw or "").strip()
    if not text.startswith("data:"):
        return None
    payload = text[5:].strip()
    if payload == "" or payload == "[DONE]":
        return None
    try:
        record = json.loads(payload)
    except json.JSONDecodeError:
        return payload
    if not isinstance(record, dict):
        return payload
    line = record.get("line")
    if line is None:
        line = record.get("message")
    if line is None:
        return None
    return str(line)


def _read_worker_logs(endpoint_id: str, worker_id: str, *, tail: int) -> str:
    """Read the historical backfill from the worker log stream, then stop.

    The route stays open for live events after the backfill. A read timeout
    ends the request once those historical lines have arrived.
    """
    limit = max(1, min(int(tail), 5000))
    url = f"{runpod_api_base()}/v2/serverless/{endpoint_id}/workers/{worker_id}/logs"
    lines: list[str] = []
    try:
        with httpx.stream(
            "GET",
            url,
            headers={"Authorization": f"Bearer {_api_key()}"},
            params={"tail": limit, "source": "container"},
            timeout=httpx.Timeout(30.0, read=8.0),
        ) as response:
            if response.status_code >= 400:
                detail = response.read().decode(errors="replace")[:500]
                raise RunpodBackendError(
                    f"Runpod GET worker logs failed ({response.status_code}): {detail}"
                )
            for raw in response.iter_lines():
                line = _sse_data_line(raw)
                if line is None:
                    continue
                lines.append(line)
                if len(lines) >= limit:
                    break
    except httpx.ReadTimeout:
        return "\n".join(lines)
    except httpx.HTTPError as exc:
        if lines:
            return "\n".join(lines)
        raise RunpodBackendError(f"Runpod worker log request failed: {exc}") from exc
    return "\n".join(lines)


def get_model_logs(model_id: str, *, tail: int = 200) -> str:
    """Recent container stdout from each active Runpod worker for this model."""
    endpoint = find_endpoint(model_id)
    if endpoint is None:
        raise RunpodBackendError(f"Runpod endpoint {synapsai_app_name(model_id)} is not deployed")
    endpoint_id = str(endpoint.get("id") or "")
    if endpoint_id == "":
        raise RunpodBackendError("Runpod endpoint did not include an id")
    sections: list[str] = []
    for worker in _list_workers(endpoint_id):
        worker_id = str(worker.get("id") or "")
        if worker_id == "":
            continue
        state = _worker_state(str(worker.get("status") or ""))
        body = _read_worker_logs(endpoint_id, worker_id, tail=tail)
        sections.append(f"----- {worker_id} ({state}) -----\n{body}")
    return "\n".join(sections)


def stop_endpoint_worker(worker_id: str) -> dict[str, Any]:
    target = str(worker_id or "").strip()
    if target == "":
        raise RunpodBackendError("worker id is required")
    endpoint_id = _WORKER_ENDPOINTS.get(target) or _find_worker_endpoint(target)
    if endpoint_id is None:
        raise RunpodBackendError(f"Runpod worker {target} was not found")
    try:
        _request("DELETE", f"/v2/serverless/{endpoint_id}/workers/{target}")
    except RunpodBackendError as exc:
        if _error_is_missing(exc):
            return {"ok": True, "missing": True, "id": target}
        raise
    return {"ok": True, "id": target, "endpoint_id": endpoint_id}


def _find_worker_endpoint(worker_id: str) -> str | None:
    for endpoint in list_endpoints():
        name = str(endpoint.get("name") or "")
        if not name.startswith("synapsai-"):
            continue
        endpoint_id = str(endpoint.get("id") or "")
        for worker in _list_workers(endpoint_id):
            if str(worker.get("id") or "") == worker_id:
                _WORKER_ENDPOINTS[worker_id] = endpoint_id
                return endpoint_id
    return None
