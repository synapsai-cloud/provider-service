"""On-demand machines for Beam replicas that stay up.

Model deployment calls ``ensure_ondemand_pool`` from ``deploy_inference_pod``.
``min=1, max=2`` reserves one machine and also deploys a serverless Pod for
the extra replica. ``min=0`` deploys Pods only and does not reserve a machine.
``BEAM_ONDEMAND_ONLY=1`` reserves the full ``max`` as on-demand machines and
skips the serverless Pod, including when ``min`` is 0.
"""

from __future__ import annotations

import time
from typing import Any

from provider_service.settings import synapsai_app_name
from provider_service.log import get_logger

logger = get_logger(logger_name="beam_capacity")

_WARM_SUFFIX = "-warm"
_CHECKPOINT_CACHE_PREFIX = "checkpoint-model-cache-"
_PROVISION_TIMEOUT_S = 15 * 60
_POLL_SECONDS = 5


def split_capacity(min_instances: int, max_instances: int) -> tuple[int, int]:
    """Return ``(persistent, burst_max)``.

    ``min=1, max=2`` is one on-demand replica plus one serverless Pod.
    ``min=0, max=2`` is serverless Pods only.
    ``BEAM_ONDEMAND_ONLY=1`` returns ``(max, 0)`` so a zero minimum still
    reserves on-demand machines and does not deploy a serverless Pod.
    """
    from provider_service.settings import beam_ondemand_only

    persistent = max(0, int(min_instances))
    maximum = max(persistent, int(max_instances), 1)
    if beam_ondemand_only():
        return maximum, 0
    return persistent, maximum - persistent


def warm_app_name(model_id: str) -> str:
    base = synapsai_app_name(model_id)
    trimmed = base[: 63 - len(_WARM_SUFFIX)].strip("-.")
    return f"{trimmed}{_WARM_SUFFIX}"


def pool_name_for(model_id: str) -> str:
    return warm_app_name(model_id)


def is_warm_app(name: str) -> bool:
    return str(name or "").endswith(_WARM_SUFFIX)


def function_name_for_app(app_name: str, function_name_from_app) -> str:
    """Map both the burst Pod and the warm Pod onto the same model."""
    name = function_name_from_app(app_name)
    if name.endswith(_WARM_SUFFIX):
        return name[: -len(_WARM_SUFFIX)]
    return name


def _gateway():
    from provider_service.providers.beam.app import _gateway as gateway

    return gateway()


def _error(message: str):
    from provider_service.providers.beam.app import BeamBackendError

    return BeamBackendError(message)


def _pool_by_name(gateway, name: str) -> Any | None:
    from beta9.clients.gateway import ListPrivatePoolsRequest

    response = gateway.list_private_pools(ListPrivatePoolsRequest())
    if not response.ok:
        raise _error(response.err_msg or f"Failed to list Beam pools while looking for {name}")
    for pool in response.pools or []:
        if pool.name == name or pool.selector == name:
            return pool
    return None


def _matching_offers(gateway, gpu: str, gpu_count: int) -> list[Any]:
    from beta9.clients.gateway import ListPoolOffersRequest, PoolConfig

    response = gateway.list_pool_offers(ListPoolOffersRequest(pool=PoolConfig(gpu=[gpu])))
    if not response.ok:
        raise _error(response.err_msg or f"Failed to list on-demand offers for {gpu}")
    needed = max(1, int(gpu_count or 1))
    offers = [
        offer
        for offer in (response.offers or [])
        if str(getattr(offer, "gpu", "") or "") == gpu
        and int(getattr(offer, "gpu_count", 0) or 1) >= needed
    ]
    offers.sort(key=lambda offer: int(getattr(offer, "hourly_cost_micros", 0) or 0))
    return offers


def _ready_machines(pool) -> int:
    if pool is None:
        return 0
    return int(getattr(pool, "ready_machine_count", 0) or 0)


def ensure_ondemand_pool(
    *,
    model_id: str,
    gpu: str,
    gpu_count: int,
    nodes: int,
    gateway=None,
) -> str:
    """Reserve ``nodes`` machines, or return the pool if they are already ready."""
    from beta9.abstractions.base.capacity import INDEFINITE_TTL, pool_config_for_offer
    from beta9.clients.gateway import LaunchPoolCapacityRequest

    count = max(1, int(nodes))
    name = pool_name_for(model_id)
    gateway = gateway or _gateway()
    current = _pool_by_name(gateway, name)
    if _ready_machines(current) >= count:
        return name
    reserved = int(getattr(current, "reserved_nodes", 0) or 0) if current is not None else 0
    if reserved >= count:
        if not _wait_until_ready(gateway, name, count):
            raise _error(f"Timed out waiting for on-demand pool {name} to become ready")
        return name

    offers = _matching_offers(gateway, gpu, gpu_count)
    if not offers:
        raise _error(
            f"No on-demand {gpu} offer has {max(1, int(gpu_count or 1))} GPU(s). "
            "Check `beam pool offers`."
        )
    offer = offers[0]
    missing = count - _ready_machines(current)
    config = pool_config_for_offer(
        offer,
        name=name,
        nodes=missing,
        ttl=INDEFINITE_TTL,
    )
    hourly = int(getattr(offer, "hourly_cost_micros", 0) or 0) / 1_000_000
    logger.info(
        "Reserving %s x%s node(s) into %s at about $%.2f/hr each",
        gpu,
        missing,
        name,
        hourly,
    )
    launched = gateway.launch_pool_capacity(
        LaunchPoolCapacityRequest(pool=config, nodes=missing)
    )
    if not launched.ok:
        raise _error(getattr(launched, "err_msg", None) or f"Failed to reserve Beam pool {name}")
    if not _wait_until_ready(gateway, name, count):
        raise _error(f"Timed out waiting for on-demand pool {name} to become ready")
    return name


def _wait_until_ready(gateway, name: str, nodes: int) -> bool:
    deadline = time.monotonic() + _PROVISION_TIMEOUT_S
    while time.monotonic() < deadline:
        pool = _pool_by_name(gateway, name)
        if _ready_machines(pool) >= nodes:
            return True
        time.sleep(_POLL_SECONDS)
    return False


def burst_storage_gb(
    model_id: str,
    *,
    has_burst_deployment: bool | None = None,
    volumes: list[Any] | None = None,
) -> float:
    """GB Beam reports for the burst Pod's volumes.

    On-demand machines are skipped: their disk is part of the machine.
    Volume ``size`` is the byte count from ``ListVolumes``. A missing burst
    deployment is 0 even if a volume is left behind.
    """
    from provider_service.settings import synapsai_app_name

    app_name = synapsai_app_name(model_id)
    if has_burst_deployment is None:
        from provider_service.providers.beam.app import find_named_deployment

        has_burst_deployment = find_named_deployment(app_name) is not None
    if not has_burst_deployment:
        return 0.0
    if volumes is None:
        volumes = _list_volumes()
    warm = warm_app_name(model_id)
    total_bytes = 0
    for volume in volumes:
        name = str(_volume_field(volume, "name") or "")
        if not _volume_is_burst(name, app_name, warm):
            continue
        total_bytes += int(_volume_field(volume, "size") or 0)
    return total_bytes / (1024 ** 3)


def _volume_field(volume: Any, field: str) -> Any:
    if isinstance(volume, dict):
        return volume.get(field)
    return getattr(volume, field, None)


def _volume_is_burst(name: str, app_name: str, warm_name: str) -> bool:
    if name == "" or name == warm_name or name.startswith(f"{warm_name}-") or name.startswith(f"{warm_name}/"):
        return False
    return _volume_owned_by(name, app_name)


def _name_belongs_to_app(name: str, app_name: str) -> bool:
    return name == app_name or name.startswith(f"{app_name}-") or name.startswith(f"{app_name}/")


def _checkpoint_cache_for(name: str, app_name: str) -> bool:
    """Beam stores a memory checkpoint at ``checkpoint-model-cache-<deployment>-<id>``.

    ``<id>`` is one lowercase token. The warm Pod's cache therefore cannot
    match the burst name, because its remainder is ``warm-<id>``.
    """
    if not name.startswith(_CHECKPOINT_CACHE_PREFIX):
        return False
    subject = name[len(_CHECKPOINT_CACHE_PREFIX):]
    if is_warm_app(app_name):
        return _checkpoint_subject_for(subject, app_name)
    warm = f"{app_name}{_WARM_SUFFIX}"
    if subject == warm or subject.startswith(f"{warm}-") or subject.startswith(f"{warm}/"):
        return False
    return _checkpoint_subject_for(subject, app_name)


def _checkpoint_subject_for(subject: str, app_name: str) -> bool:
    if subject == app_name:
        return True
    marker = f"{app_name}-"
    if not subject.startswith(marker):
        return False
    suffix = subject[len(marker):]
    return suffix.isascii() and suffix.isalnum() and suffix.lower() == suffix


def _volume_owned_by(name: str, app_name: str) -> bool:
    """True when this volume belongs to the deployment name.

    That is the 9P volume and Beam's checkpoint model cache. The warm Pod
    is named ``<burst>-warm``. Deleting the burst deployment must leave
    that sibling's volumes alone.
    """
    if name == "" or app_name == "":
        return False
    if _checkpoint_cache_for(name, app_name):
        return True
    if is_warm_app(app_name):
        return _name_belongs_to_app(name, app_name)
    warm = f"{app_name}{_WARM_SUFFIX}"
    if name == warm or name.startswith(f"{warm}-") or name.startswith(f"{warm}/"):
        return False
    return _name_belongs_to_app(name, app_name)


def volume_names_for_app(app_name: str, volumes: list[Any]) -> list[str]:
    """Volume names Beam created for this deployment, including checkpoint caches."""
    found: list[str] = []
    for volume in volumes:
        name = str(_volume_field(volume, "name") or "")
        if _volume_owned_by(name, app_name) and name not in found:
            found.append(name)
    return found


def release_app_volumes(app_name: str, volumes: list[Any] | None = None) -> None:
    """Delete the volumes for one deployment, including checkpoint caches. Missing volumes are fine."""
    from beta9.channel import ServiceClient
    from beta9.clients.volume import DeleteVolumeRequest

    from provider_service.providers.beam.app import _config_context, _require_beam

    names = volume_names_for_app(app_name, volumes if volumes is not None else _list_volumes())
    if not names:
        return
    _require_beam()
    client = ServiceClient(config=_config_context())
    errors: list[str] = []
    try:
        for name in names:
            response = client.volume.delete_volume(DeleteVolumeRequest(name=name))
            if response.ok:
                logger.info("Deleted Beam volume %s", name)
                continue
            message = str(getattr(response, "err_msg", "") or "")
            if "not found" in message.lower():
                continue
            logger.error("Failed to delete Beam volume %s: %s", name, message or "Beam rejected the delete")
            errors.append(message or f"Failed to delete Beam volume {name}")
    finally:
        client.close()
    if errors:
        raise _error("; ".join(errors))


def _list_volumes() -> list[Any]:
    from beta9.channel import ServiceClient
    from beta9.clients.volume import ListVolumesRequest

    from provider_service.providers.beam.app import _config_context, _require_beam

    _require_beam()
    client = ServiceClient(config=_config_context())
    try:
        response = client.volume.list_volumes(ListVolumesRequest())
    finally:
        client.close()
    if not response.ok:
        raise _error(response.err_msg or "Failed to list Beam volumes")
    return list(response.volumes or [])


def release_ondemand_pool(model_id: str, gateway=None) -> None:
    """Stop billing for this model's reserved machines. Missing pools are fine."""
    from beta9.clients.gateway import DeletePoolRequest

    name = pool_name_for(model_id)
    gateway = gateway or _gateway()
    current = _pool_by_name(gateway, name)
    if current is None:
        return
    response = gateway.delete_pool(DeletePoolRequest(name=name))
    if not response.ok:
        message = str(getattr(response, "err_msg", "") or "").lower()
        if "not found" in message:
            return
        raise _error(getattr(response, "err_msg", None) or f"Failed to release Beam pool {name}")
    logger.info("Released on-demand pool %s", name)
