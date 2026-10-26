"""Shared shapes for inference-provider adapters.

The orchestrator talks to every backend through these records. Provider
specific work stays in ``providers/<name>/``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

INSTANCE_STATUS_READY = "ready"
INSTANCE_STATUS_RESUMING = "resuming"
INSTANCE_STATUS_SLEEPING = "sleeping"
INSTANCE_STATUS_BUILDING = "building"
INSTANCE_STATUS_FAILED = "failed"
INSTANCE_STATUS_STOPPED = "stopped"


@dataclass(frozen=True)
class ProviderInstance:
    """Standardized instance record returned by every inference provider."""

    id: str
    name: str
    funcname: str
    state: str
    status: str = INSTANCE_STATUS_READY
    warm: bool = False
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "funcname": self.funcname,
            "state": self.state,
            "status": self.status,
            "warm": self.warm,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class ModelStatusSummary:
    """Standardized model status and serving instances from a provider."""

    status: str
    active_instances: int
    serving_instances: list[ProviderInstance]


@dataclass(frozen=True)
class InstanceHealth:
    """One always-on replica. Burst Pods that scale to zero are omitted."""

    instance_id: str
    state: str
    url: str = ""
    detail: str = ""


@dataclass(frozen=True)
class InferenceRoute:
    """Where the next request should be sent."""

    url: str
    role: str


@dataclass(frozen=True)
class ApplicationLogs:
    """Stdout from a running custom-image replica since the previous read.

    ``cursor`` is opaque. The caller stores it and passes it back so the
    provider can return only lines that were not already collected.
    """

    lines: tuple[str, ...] = ()
    cursor: str = ""


def tail_log_text(text: str, tail: int) -> str:
    """Keep the last ``tail`` lines. ``tail <= 0`` returns the full text."""
    lines = str(text or "").splitlines()
    if tail > 0 and len(lines) > tail:
        lines = lines[-tail:]
    return "\n".join(lines)


def function_key(name: str | None) -> str:
    return str(name or "").strip().casefold()


def model_function_keys(model_id: str) -> set[str]:
    """Every name a provider pod row may use for this model.

    Modal, Runpod, and Beam publish a sanitized app name. Long model ids are
    truncated to fit that name, so the pod function is not always ``model_id``.
    """
    from provider_service.settings import function_name_from_app, synapsai_app_name
    from provider_service.inferx_client import inferx_function_name

    raw = str(model_id or "").strip()
    if raw == "":
        return set()
    keys = {raw}
    try:
        keys.add(inferx_function_name(raw))
    except Exception:
        pass
    try:
        keys.add(function_name_from_app(synapsai_app_name(raw)))
    except Exception:
        pass
    return {key for key in keys if str(key).strip()}


def matching_instance(
    rows: list[dict[str, Any]],
    function_name: str,
    model_inference_id: str,
    *,
    model_id: str = "",
) -> dict[str, Any] | None:
    target = str(model_inference_id or "").strip()
    allowed_keys = {function_key(function_name)} if str(function_name or "").strip() else set()
    if model_id:
        allowed_keys.update(function_key(key) for key in model_function_keys(model_id))
    for row in rows:
        funcname = (row.get("funcname") or "").strip()
        if funcname != "" and allowed_keys and function_key(funcname) not in allowed_keys:
            continue
        if target in {
            (row.get("id") or "").strip(),
            (row.get("name") or "").strip(),
        }:
            return row
    return None


def status_from_instance_statuses(statuses: list[str]) -> tuple[str, int]:
    """Map canonical instance statuses onto a model status and serving count."""
    normalized = [str(status or "").strip().lower() for status in statuses]
    serving = sum(1 for status in normalized if status == INSTANCE_STATUS_READY)
    if serving:
        return "ready", serving
    if any(status == INSTANCE_STATUS_RESUMING for status in normalized):
        return "ready", 0
    if any(status == INSTANCE_STATUS_SLEEPING for status in normalized):
        return "sleeping", 0
    if any(status == INSTANCE_STATUS_FAILED for status in normalized):
        return "failed", 0
    if any(status == INSTANCE_STATUS_BUILDING for status in normalized):
        return "building", 0
    return "sleeping", 0


def assign_warm_by_min_instances(
    instances: list[ProviderInstance],
    min_instances: int,
) -> list[ProviderInstance]:
    """Mark up to ``min_instances`` serving replicas as warm; extras are burst."""
    minimum = max(0, int(min_instances or 0))
    if minimum == 0 and any(inst.warm for inst in instances):
        return instances
    warm_remaining = minimum
    updated: list[ProviderInstance] = []
    for inst in instances:
        if inst.status == INSTANCE_STATUS_READY and warm_remaining > 0:
            warm_remaining -= 1
            updated.append(replace(inst, warm=True))
        else:
            updated.append(replace(inst, warm=False))
    return updated


def summarize_provider_instances(
    backend: Any,
    pods: list[dict[str, Any]],
    *,
    min_instances: int = 0,
) -> ModelStatusSummary:
    """Delegate instance normalization and status calculation to the provider."""
    summarizer = getattr(backend, "summarize_instances", None)
    if callable(summarizer):
        return summarizer(pods, min_instances=min_instances)
    backend_name = str(getattr(backend, "backend_name", "") or "").strip().lower()
    if not backend_name and pods:
        backend_name = str(pods[0].get("backend") or "").strip().lower()
    if backend_name == "beam":
        from provider_service.providers.beam.app import summarize_pod_instances

        return summarize_pod_instances(pods, min_instances=min_instances)
    if backend_name == "modal":
        from provider_service.providers.modal.app import summarize_server_instances

        return summarize_server_instances(pods, min_instances=min_instances)
    if backend_name == "runpod":
        from provider_service.providers.runpod.app import summarize_endpoint_instances

        return summarize_endpoint_instances(pods, min_instances=min_instances)
    from provider_service.providers.inferx.app import summarize_pod_instances as summarize_inferx

    return summarize_inferx(pods, min_instances=min_instances)


def collect_model_summaries(
    backend: Any,
    rows: list[dict[str, Any]],
    models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match provider rows to the caller's models and return one summary each.

    Each item is ``{"key", "matched", "skip", "summary"}``. ``summary`` is a
    ``ModelStatusSummary``. Unmatched rows are returned with ``matched`` false
    and ``key`` set to the provider function name.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        funcname = str(row.get("funcname") or row.get("name") or "").strip()
        if funcname == "":
            continue
        buckets.setdefault(function_key(funcname), []).append(row)
    claimed: set[int] = set()

    def take(aliases: list[str]) -> list[dict[str, Any]]:
        pods: list[dict[str, Any]] = []
        for alias in aliases:
            for row in buckets.get(function_key(alias), []):
                marker = id(row)
                if marker in claimed:
                    continue
                claimed.add(marker)
                pods.append(row)
        return pods

    results: list[dict[str, Any]] = []
    for spec in models:
        key = str(spec.get("key") or "")
        aliases = [str(alias) for alias in (spec.get("aliases") or []) if str(alias).strip()]
        if key and key not in aliases:
            aliases.append(key)
        pods = take(aliases)
        if spec.get("skip"):
            results.append({"key": key, "matched": True, "skip": True, "summary": None})
            continue
        summary = summarize_provider_instances(
            backend,
            pods,
            min_instances=int(spec.get("min_instances") or 0),
        )
        results.append({"key": key, "matched": True, "skip": False, "summary": summary})

    for grouped in buckets.values():
        leftover = [row for row in grouped if id(row) not in claimed]
        if not leftover:
            continue
        funcname = str(leftover[0].get("funcname") or leftover[0].get("name") or "")
        summary = summarize_provider_instances(backend, leftover, min_instances=0)
        results.append({"key": funcname, "matched": False, "skip": False, "summary": summary})
    return results
