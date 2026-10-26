"""InferX pod state classification and instance normalization."""

from __future__ import annotations

from typing import Any

from provider_service.providers.schema import (
    INSTANCE_STATUS_BUILDING,
    INSTANCE_STATUS_FAILED,
    INSTANCE_STATUS_READY,
    INSTANCE_STATUS_RESUMING,
    INSTANCE_STATUS_SLEEPING,
    INSTANCE_STATUS_STOPPED,
    ModelStatusSummary,
    ProviderInstance,
    assign_warm_by_min_instances,
    function_key,
    status_from_instance_statuses,
)
from provider_service.inferx_client import inferx_fprevision_from_pod_name

# Pods in these states can serve inference and are billed as loaded footprints.
_SERVING_POD_STATES = {"ready", "running", "working", "idle"}
# Hibernated pods are deployed but not serving.
_SLEEPING_POD_STATES = {"standby", "memhibernated", "snapshoted"}
# A standby pod waking to serve. The function is already deployed.
_RESUMING_POD_STATES = {"resuming", "resumedone"}
# Pods leaving the node. They do not keep the model awake.
_IGNORED_POD_STATES = {"", "terminating", "terminated", "cleanup", "deleted"}
_FAILED_POD_STATES = {"failed"}
# InferX snapshot pods move Init → PullingImage → Creating → Loading → Snapshoting.
# Loading here is snapshot creation, not a SynapsAI deployment build.
_SNAPSHOT_ACTIVE_POD_STATES = {
    "init",
    "pullingimage",
    "creating",
    "created",
    "loading",
    "snapshoting",
}
# Restore brings a snapshot back to standby. Same deployed-but-not-serving idea.
_RESTORE_ACTIVE_POD_STATES = {
    "init",
    "pullingimage",
    "creating",
    "created",
    "loading",
    "loadingtimeout",
    "restoring",
}


def _nested(pod: dict, *keys: str) -> Any:
    current: Any = pod
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def pod_state(pod: dict[str, Any]) -> str:
    direct = pod.get("state")
    if isinstance(direct, str) and direct.strip() != "":
        return direct.strip().lower()
    nested = _nested(pod, "object", "status", "state")
    return str(nested or "").strip().lower()


def raw_pod_state(pod: dict[str, Any]) -> str:
    direct = pod.get("state")
    if isinstance(direct, str) and direct.strip() != "":
        return direct.strip()
    nested = _nested(pod, "object", "status", "state")
    return str(nested or "").strip()


def pod_create_type(pod: dict[str, Any]) -> str:
    direct = pod.get("create_type")
    if isinstance(direct, str) and direct.strip() != "":
        return direct.strip().lower()
    nested = _nested(pod, "object", "spec", "create_type")
    return str(nested or "").strip().lower()


def pod_funcname(pod: dict[str, Any]) -> str:
    direct = pod.get("funcname")
    if isinstance(direct, str) and direct.strip() != "":
        return direct.strip()
    nested = _nested(pod, "object", "spec", "funcname")
    if isinstance(nested, str) and nested.strip() != "":
        return nested.strip()
    return str(pod.get("name") or "").strip()


def pod_status(pod: dict[str, Any]) -> str:
    """Translate an InferX pod row into a canonical provider instance status."""
    explicit = str(pod.get("status") or "").strip().lower()
    if explicit in {
        INSTANCE_STATUS_READY,
        INSTANCE_STATUS_RESUMING,
        INSTANCE_STATUS_SLEEPING,
        INSTANCE_STATUS_BUILDING,
        INSTANCE_STATUS_FAILED,
        INSTANCE_STATUS_STOPPED,
    }:
        return explicit
    state = pod_state(pod)
    create_type = pod_create_type(pod)
    if state in _SERVING_POD_STATES:
        return INSTANCE_STATUS_READY
    if state in _RESUMING_POD_STATES:
        return INSTANCE_STATUS_RESUMING
    if state in _SLEEPING_POD_STATES:
        return INSTANCE_STATUS_SLEEPING
    if create_type == "restore" and state in _RESTORE_ACTIVE_POD_STATES:
        return INSTANCE_STATUS_SLEEPING
    if (create_type == "snapshot" and state in _SNAPSHOT_ACTIVE_POD_STATES) or state in {
        "snapshoting",
        "restoring",
    }:
        return INSTANCE_STATUS_SLEEPING
    if state in _FAILED_POD_STATES:
        return INSTANCE_STATUS_FAILED
    if state in _IGNORED_POD_STATES:
        return INSTANCE_STATUS_STOPPED
    return INSTANCE_STATUS_BUILDING


def status_for_pods(pods: list[dict[str, Any]]) -> tuple[str, int]:
    """Map InferX pod states to a model status and a serving-instance count.

    Ready and working pods keep the model ready. Standby means sleeping.
    A Snapshot or Restore pod that is still loading is an already deployed
    function creating or applying a snapshot, so it stays sleeping instead of
    flipping back to building. Other in-progress states, such as a normal pod
    still creating, mean the model is building.
    """
    return status_from_instance_statuses([pod_status(pod) for pod in pods])


def normalize_pod_instance(
    pod: dict[str, Any],
    *,
    warm: bool | None = None,
) -> ProviderInstance:
    funcname = pod_funcname(pod)
    name = str(pod.get("name") or pod.get("id") or funcname)
    pod_id = str(pod.get("id") or name or funcname)
    resolved_warm = bool(pod.get("warm")) if warm is None else bool(warm)
    return ProviderInstance(
        id=pod_id,
        name=name,
        funcname=funcname,
        state=raw_pod_state(pod),
        status=pod_status(pod),
        warm=resolved_warm,
        backend="inferx",
    )


def normalize_pod_instances(
    rows: list[dict[str, Any]],
    *,
    min_instances_by_function: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Attach canonical ``status``, ``warm``, and ``backend`` fields to InferX pod rows."""
    minimums = {
        function_key(key): max(0, int(val or 0))
        for key, val in (min_instances_by_function or {}).items()
    }
    serving_seen: dict[str, int] = {}
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        funcname = pod_funcname(copied)
        fkey = function_key(funcname)
        status = pod_status(copied)
        if isinstance(copied.get("warm"), bool):
            warm = copied["warm"]
        elif status == INSTANCE_STATUS_READY:
            seen = serving_seen.get(fkey, 0)
            serving_seen[fkey] = seen + 1
            warm = seen < minimums.get(fkey, 0)
        else:
            warm = False
        fprevision = str(copied.get("fprevision") or "").strip()
        if fprevision == "":
            fprevision = inferx_fprevision_from_pod_name(str(copied.get("name") or ""))
            if fprevision != "":
                copied["fprevision"] = fprevision
        instance = normalize_pod_instance(copied, warm=warm)
        copied.update(instance.to_dict())
        normalized_rows.append(copied)
    return normalized_rows


def summarize_pod_instances(
    pods: list[dict[str, Any]],
    *,
    min_instances: int = 0,
) -> ModelStatusSummary:
    instances = [
        normalize_pod_instance(
            pod,
            warm=pod["warm"] if isinstance(pod.get("warm"), bool) else None,
        )
        for pod in pods
    ]
    instances = assign_warm_by_min_instances(instances, min_instances)
    status, active = status_from_instance_statuses([inst.status for inst in instances])
    serving = [inst for inst in instances if inst.status == INSTANCE_STATUS_READY]
    return ModelStatusSummary(
        status=status,
        active_instances=active,
        serving_instances=serving,
    )
