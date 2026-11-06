"""Switch the inference control plane between InferX, Modal, Runpod, and Beam.

``INFERENCE_BACKEND`` must be set. It is one provider (``beam``) or a route list
(``warm=runpod,burst=modal``). Callers that name a backend on the request do
not use this value. Modal, Runpod, and Beam start the command and image the
caller sends. InferX keeps its own function image.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from provider_service.inferx_client import inferx_function_name
from provider_service.providers.beam.gpus import BEAM_ONDEMAND_GPUS, BEAM_POD_GPUS
from provider_service.providers.modal.gpus import MODAL_GPUS
from provider_service.providers.runpod.gpus import RUNPOD_GPU_POOLS

INFERENCE_BACKENDS = ("inferx", "modal", "runpod", "beam")
_V3_BACKENDS = frozenset({"modal", "runpod", "beam"})
_BACKEND_LABELS = {
    "inferx": "InferX",
    "modal": "Modal",
    "runpod": "Runpod",
    "beam": "Beam",
}

DEFAULT_LISTEN_PORT = 8000
MODAL_APP_PREFIX = "synapsai-"
MODAL_SERVER_NAME = "Inference"


class InferenceBackendError(Exception):
    pass


def _first_nonempty(*values: str | None) -> str:
    for value in values:
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def inference_backend_setting() -> str:
    """Return ``INFERENCE_BACKEND``. An empty value is an error."""
    raw = _first_nonempty(os.getenv("INFERENCE_BACKEND"))
    if raw.startswith("="):
        raw = raw[1:].strip()
    if raw == "":
        raise InferenceBackendError(
            "INFERENCE_BACKEND must be set to one provider "
            "(inferx, modal, runpod, beam) or a route list "
            "such as warm=runpod,burst=modal"
        )
    return raw


def resolve_inference_backend(name: str | None = None) -> str:
    """Return one known provider.

    ``name`` is that provider. With no name, ``INFERENCE_BACKEND`` must itself
    be a single provider. A route list is not a provider name.
    """
    if name is not None and str(name).strip() != "":
        resolved = str(name).strip().lower()
    else:
        resolved = inference_backend_setting().lower()
    if resolved not in INFERENCE_BACKENDS:
        allowed = ", ".join(INFERENCE_BACKENDS)
        raise InferenceBackendError(
            "INFERENCE_BACKEND must name one provider "
            f"({allowed}) or a route list such as warm=runpod,burst=modal "
            f"(got {resolved!r})"
        )
    return resolved


@dataclass(frozen=True)
class BackendRoute:
    """One scaling rule. ``op`` is ``=``, ``>=``, ``>``, ``<=``, ``<``, ``warm``, or ``burst``."""

    op: str
    value: int
    backend: str


_ROUTE_RE = re.compile(
    r"^(?:min_workers|min_instances)\s*(>=|<=|>|<|=)\s*(\d+)\s*[=:]\s*([a-z0-9_-]+)$",
    re.IGNORECASE,
)
_ROLE_ROUTE_RE = re.compile(r"^(warm|burst)\s*[=:]\s*([a-z0-9_-]+)$", re.IGNORECASE)


def parse_backend_routes(raw: str | None = None) -> list[BackendRoute]:
    """Parse ``INFERENCE_BACKEND`` when it is a route list.

    A single provider such as ``beam`` has no routes. Rules are comma,
    semicolon, or newline separated and evaluated in order.
    ``min_workers>=1=runpod,min_workers=0=modal`` places warm models on Runpod
    and scale-to-zero models on Modal. ``warm=runpod,burst=modal`` keeps one
    model's always-on replicas on Runpod and its scale-out replicas on Modal.
    A scaling policy that matches no rule is an error.
    """
    if raw is None:
        text = inference_backend_setting()
    else:
        text = str(raw or "").strip()
        if text.startswith("="):
            text = text[1:].strip()
    if text == "" or text.lower() in INFERENCE_BACKENDS:
        return []
    routes: list[BackendRoute] = []
    for part in re.split(r"[,;\n]+", text):
        piece = part.strip()
        if piece == "":
            continue
        if piece.startswith("="):
            piece = piece[1:].strip()
        role_match = _ROLE_ROUTE_RE.fullmatch(piece)
        if role_match is not None:
            backend = role_match.group(2).lower()
            _require_backend_name(backend, piece)
            routes.append(BackendRoute(op=role_match.group(1).lower(), value=0, backend=backend))
            continue
        match = _ROUTE_RE.fullmatch(piece)
        if match is None:
            raise InferenceBackendError(
                "INFERENCE_BACKEND is one provider, or rules like "
                "min_workers>=1=runpod,min_workers=0=modal or "
                f"warm=runpod,burst=modal (got {piece!r})"
            )
        backend = match.group(3).lower()
        _require_backend_name(backend, piece)
        routes.append(BackendRoute(op=match.group(1), value=int(match.group(2)), backend=backend))
    return routes


def _require_backend_name(backend: str, piece: str) -> None:
    if backend not in INFERENCE_BACKENDS:
        allowed = ", ".join(INFERENCE_BACKENDS)
        raise InferenceBackendError(
            f"INFERENCE_BACKEND provider must be one of: {allowed} (got {piece!r})"
        )


def _route_matches(route: BackendRoute, min_workers: int) -> bool:
    if route.op in {"warm", "burst"}:
        return False
    if route.op == "=":
        return min_workers == route.value
    if route.op == ">=":
        return min_workers >= route.value
    if route.op == ">":
        return min_workers > route.value
    if route.op == "<=":
        return min_workers <= route.value
    if route.op == "<":
        return min_workers < route.value
    return False


def select_inference_backend(*, min_workers: int, routes: str | None = None) -> str:
    """Pick a provider from ``INFERENCE_BACKEND``.

    A single provider always wins. ``min_workers`` rules are evaluated in
    order. ``warm`` and ``burst`` apply when those are the only rules.
    A policy that matches nothing raises.
    """
    workers = int(min_workers or 0)
    parsed = parse_backend_routes(routes)
    for route in parsed:
        if _route_matches(route, workers):
            return route.backend
    if not parsed:
        if routes is not None and str(routes).strip() != "":
            return resolve_inference_backend(str(routes).strip())
        return resolve_inference_backend()
    warm = next((route.backend for route in parsed if route.op == "warm"), None)
    burst = next((route.backend for route in parsed if route.op == "burst"), None)
    if workers > 0 and warm:
        return warm
    if workers <= 0 and burst:
        return burst
    if warm or burst:
        return warm or burst
    raise InferenceBackendError("INFERENCE_BACKEND has no provider for this scaling policy")


def backend_names_in_use() -> list[str]:
    """Providers named by ``INFERENCE_BACKEND``. Nothing else is added."""
    routes = parse_backend_routes()
    if not routes:
        return [resolve_inference_backend()]
    names: list[str] = []
    for route in routes:
        if route.backend not in names:
            names.append(route.backend)
    return names


def uses_inference_v3(backend: Any) -> bool:
    name = getattr(backend, "backend_name", None)
    if not name:
        name = resolve_inference_backend()
    return name in _V3_BACKENDS


def backend_label(backend: Any = None) -> str:
    name = getattr(backend, "backend_name", None) if backend is not None else None
    if not name:
        name = resolve_inference_backend()
    return _BACKEND_LABELS.get(name, name or "provider")


def synapsai_app_name(model_id: str) -> str:
    """Stable lowercase name shared by Modal apps and Runpod endpoints."""
    raw = inferx_function_name(model_id).lower()
    cleaned = re.sub(r"[^a-z0-9.-]+", "-", raw).strip("-.")
    if cleaned == "":
        raise InferenceBackendError(f"model_id {model_id!r} cannot be used as an app name")
    name = f"{MODAL_APP_PREFIX}{cleaned}"
    return name[:63].strip("-.")


def modal_app_name(model_id: str) -> str:
    """Modal App name for a SynapsAI model. Stable and lowercase."""
    return synapsai_app_name(model_id)


def function_name_from_app(app_name: str) -> str:
    name = str(app_name or "").strip()
    if name.startswith(MODAL_APP_PREFIX):
        return name[len(MODAL_APP_PREFIX):]
    return name


def cuda_visible_devices_value(gpu_count: int) -> str:
    """Logical CUDA ids for a container that was given ``gpu_count`` GPUs.

    Providers expose those GPUs to the process as ``0..N-1``. Setting this
    explicitly avoids a load failure when the host leaves ``CUDA_VISIBLE_DEVICES``
    unset.
    """
    count = int(gpu_count or 1)
    if count < 1:
        raise InferenceBackendError("gpu_count must be a positive integer")
    return ",".join(str(index) for index in range(count))


def modal_gpu(vram_mb: int, gpu_count: int = 1) -> str:
    """Map a VRAM request onto a Modal GPU string (``L4``, ``H100:2``, ...).

    ``vram_mb`` is the memory one GPU must provide. ``gpu_count`` greater than
    1 uses Modal's ``TYPE:N`` form.
    """
    count = int(gpu_count or 1)
    if count <= 0:
        raise InferenceBackendError("gpu_count must be a positive integer")
    override = _first_nonempty(os.getenv("MODAL_GPU"))
    if override != "":
        if count > 1 and ":" not in override:
            return f"{override}:{count}"
        return override
    chosen = gpu_card("modal", vram_mb).name
    if count > 1:
        return f"{chosen}:{count}"
    return chosen


def modal_proxy_bearer() -> str:
    """Proxy token Modal Servers expect as ``Authorization: Bearer id.secret``.

    These are workspace proxy tokens, not the Modal SDK token pair.
    https://modal.com/docs/guide/webhook-proxy-auth
    """
    token_id = _first_nonempty(os.getenv("MODAL_PROXY_TOKEN_ID"))
    token_secret = _first_nonempty(os.getenv("MODAL_PROXY_TOKEN_SECRET"))
    if token_id == "" or token_secret == "":
        return ""
    return f"{token_id}.{token_secret}"


def modal_server_unauthenticated() -> bool:
    raw = _first_nonempty(os.getenv("MODAL_SERVER_UNAUTHENTICATED"), "0")
    return raw == "1"


def modal_environment_name() -> str | None:
    name = _first_nonempty(os.getenv("MODAL_ENVIRONMENT"))
    return name or None


def runpod_api_key() -> str:
    return _first_nonempty(os.getenv("RUNPOD_API_KEY"))


def runpod_api_base() -> str:
    return _first_nonempty(os.getenv("RUNPOD_API_BASE"), "https://api.runpod.io").rstrip("/")


def runpod_gpu_pool(
    vram_mb: int,
    gpu_count: int = 1,
    *,
    by_availability: bool = False,
) -> tuple[str, int]:
    """Map a VRAM request onto a Runpod serverless pool id and GPU count.

    ``by_availability`` keeps only pools with US secure-cloud serverless
    capacity for ``gpu_count``. The cheapest pool that still covers
    ``vram_mb`` is the one returned.
    """
    count = int(gpu_count or 1)
    if count <= 0:
        raise InferenceBackendError("gpu_count must be a positive integer")
    override = _first_nonempty(os.getenv("RUNPOD_GPU_POOL"))
    if override != "":
        return override, count
    return gpu_card("runpod", vram_mb, by_availability=by_availability, gpu_count=count).name, count


@dataclass(frozen=True)
class GpuCard:
    """One provider GPU. ``usd_per_hour`` is the provider list price for a single card."""

    name: str
    vram_mb: int
    usd_per_hour: float


def gpu_catalog(
    backend: str,
    *,
    warm: bool = False,
    by_availability: bool = False,
    gpu_count: int = 1,
) -> tuple:
    """Card rows for ``backend``. Burst pods are the default. ``warm`` is Beam's reserved list.

    ``by_availability`` applies to Runpod. It replaces the static pool list
    with the pools that have US serverless capacity for ``gpu_count``.
    """
    name = str(backend or "").strip().lower()
    if name == "beam":
        return BEAM_ONDEMAND_GPUS if warm else BEAM_POD_GPUS
    if name == "modal":
        return MODAL_GPUS
    if name == "runpod":
        if by_availability:
            from provider_service.providers.runpod.catalog import pools_in_stock

            return pools_in_stock(gpu_count)
        return RUNPOD_GPU_POOLS
    raise InferenceBackendError(
        f"{name} has no GPU card prices. "
        "Hourly inference price is the full card on beam, modal, or runpod."
    )


def _card_price(row: tuple) -> float:
    if len(row) >= 3:
        return float(row[2])
    for item in BEAM_ONDEMAND_GPUS:
        if item[1] == row[1]:
            return float(item[2])
    raise InferenceBackendError(f"No hourly price for GPU {row[1]}")


def _fit_card(catalog: tuple, needed_mb: int, *, backend: str) -> tuple:
    fitting = [row for row in catalog if needed_mb <= row[0]]
    if not fitting:
        largest = max(catalog, key=lambda row: row[0])
        raise InferenceBackendError(
            f"{backend} GPUs cover up to {largest[0] // 1024} GB each ({largest[1]}). "
            "Split the model across gpu_count instead of requesting more VRAM on one GPU."
        )
    fitting.sort(key=lambda row: (row[0], _card_price(row)))
    return fitting[0]


def gpu_card(
    backend: str,
    vram_mb: int,
    *,
    warm: bool = False,
    by_availability: bool = False,
    gpu_count: int = 1,
) -> GpuCard:
    """Smallest priced card on ``backend`` that covers ``vram_mb`` for one GPU.

    Same-VRAM ties take the cheaper list price. Nothing is clamped onto a
    smaller card: the caller splits the model across ``gpu_count``.
    ``by_availability`` limits Runpod to pools with live US serverless capacity.
    """
    name = str(backend or "").strip().lower()
    needed = int(vram_mb or 0)
    if needed <= 0:
        raise InferenceBackendError("vram_mb must be a positive integer")
    catalog = gpu_catalog(name, warm=warm, by_availability=by_availability, gpu_count=gpu_count)
    if not catalog:
        raise InferenceBackendError(
            "Runpod has no serverless GPU pool with capacity in the US right now."
            if name == "runpod"
            else f"{name} has no GPU cards."
        )
    row = _fit_card(catalog, needed, backend=name)
    return GpuCard(name=row[1], vram_mb=row[0], usd_per_hour=_card_price(row))


@dataclass(frozen=True)
class GpuPlan:
    """How many whole cards a memory total needs, and the card each GPU is."""

    gpu_count: int
    vram_mb: int
    card: GpuCard


def plan_gpus(
    total_vram_mb: int,
    backend: str,
    *,
    warm: bool = False,
    by_availability: bool = False,
) -> GpuPlan:
    """Pick ``gpu_count`` so each GPU's share fits a card on ``backend``.

    One GPU when a card covers the total. Otherwise the count is the smallest
    split whose per-GPU share fits the largest card, and that share selects
    the cheapest fitting card. ``by_availability`` limits Runpod to pools
    with live US serverless capacity at that GPU count.
    """
    name = str(backend or "").strip().lower()
    total = int(total_vram_mb or 0)
    if total <= 0:
        raise InferenceBackendError("vram_mb must be a positive integer")
    live = by_availability and name == "runpod"
    count = 1
    saw_cards = False
    largest_seen = 0
    largest_name = ""
    while count <= 64:
        catalog = gpu_catalog(name, warm=warm, by_availability=live, gpu_count=count)
        if not catalog:
            break
        saw_cards = True
        biggest = max(catalog, key=lambda row: row[0])
        if biggest[0] > largest_seen:
            largest_seen = biggest[0]
            largest_name = biggest[1]
        per_gpu = (total + count - 1) // count
        if per_gpu <= biggest[0]:
            row = _fit_card(catalog, per_gpu, backend=name)
            return GpuPlan(
                gpu_count=count,
                vram_mb=per_gpu,
                card=GpuCard(name=row[1], vram_mb=row[0], usd_per_hour=_card_price(row)),
            )
        count += 1
    if live and not saw_cards:
        raise InferenceBackendError(
            "Runpod has no serverless GPU pool with capacity in the US right now."
        )
    if live and largest_seen:
        raise InferenceBackendError(
            f"Runpod GPUs with capacity cover up to {largest_seen // 1024} GB each ({largest_name}). "
            "Split the model across gpu_count instead of requesting more VRAM on one GPU."
        )
    raise InferenceBackendError(
        f"{name} cannot cover {total // 1024} GB even split across 64 GPUs."
    )


def beam_api_token() -> str:
    return _first_nonempty(os.getenv("BEAM_TOKEN"))


def beam_server_unauthenticated() -> bool:
    raw = _first_nonempty(os.getenv("BEAM_SERVER_UNAUTHENTICATED"), "0")
    return raw == "1"


def beam_checkpoint_enabled() -> bool:
    raw = _first_nonempty(os.getenv("BEAM_CHECKPOINT_ENABLED"), "1")
    return raw == "1"


def beam_ondemand_only() -> bool:
    """When true, every replica is a reserved on-demand machine.

    ``min_instances`` 0 still reserves ``max_instances`` machines. Nothing
    is placed on a serverless Pod.
    """
    raw = _first_nonempty(os.getenv("BEAM_ONDEMAND_ONLY"), "0")
    return raw == "1"


def beam_gpu(vram_mb: int, gpu_count: int = 1) -> str:
    """Map one GPU's VRAM onto a serverless Beam GPU (``RTX4090``, ``RTX5090``, ``H100``).

    ``gpu_count`` is not folded into the name. The Pod requests that many of
    the chosen type on one machine.
    """
    count = int(gpu_count or 1)
    if count <= 0:
        raise InferenceBackendError("gpu_count must be a positive integer")
    override = _first_nonempty(os.getenv("BEAM_GPU"))
    if override != "":
        return override
    needed = int(vram_mb or 0)
    if needed <= 0:
        raise InferenceBackendError("vram_mb must be a positive integer")
    for limit, name, *_rest in BEAM_POD_GPUS:
        if needed <= limit:
            return name
    raise InferenceBackendError(
        "Beam serverless GPUs cover up to 80 GB each (H100). "
        "Split the model across gpu_count instead of requesting more VRAM on one GPU."
    )


def beam_ondemand_gpu(vram_mb: int, gpu_count: int = 1) -> str:
    """Map one GPU's VRAM onto a reserved on-demand machine.

    Picks the smallest card that fits. Two cards with the same VRAM use the
    cheaper published rate. ``BEAM_ONDEMAND_GPU`` overrides the choice.
    """
    count = int(gpu_count or 1)
    if count <= 0:
        raise InferenceBackendError("gpu_count must be a positive integer")
    override = _first_nonempty(os.getenv("BEAM_ONDEMAND_GPU"))
    if override != "":
        return override
    return gpu_card("beam", vram_mb, warm=True).name


def beam_pod_cpu() -> float:
    raw = _first_nonempty(os.getenv("BEAM_POD_CPU"), "4")
    try:
        return float(raw)
    except ValueError as exc:
        raise InferenceBackendError("BEAM_POD_CPU must be a number") from exc


# Burst pods pay for host RAM by the GiB. Anything smaller than this still
# reserves 16 GiB, which is the published 4090 pod example.
BEAM_MIN_POD_MEMORY_GIB = 16


def beam_pod_ram_gib(vram_mb: int, gpu_count: int = 1) -> int:
    """Host RAM in GiB: requested VRAM across the pod, and at least 16 GiB."""
    requested = max(1, (int(vram_mb or 0) * int(gpu_count or 1)) // 1024)
    return max(BEAM_MIN_POD_MEMORY_GIB, requested)


def beam_pod_memory(vram_mb: int, gpu_count: int = 1) -> str:
    """Host RAM for a Beam pod. ``BEAM_POD_MEMORY`` overrides the 16 GiB minimum."""
    override = _first_nonempty(os.getenv("BEAM_POD_MEMORY"))
    if override != "":
        return override
    return f"{beam_pod_ram_gib(vram_mb, gpu_count)}Gi"


def beam_registry_secret_names(image: str) -> list[str] | None:
    """Workspace secret names for pulling a private registry image on Beam."""
    if not str(image).startswith("ghcr.io/"):
        return None
    username, password = registry_credentials()
    if username == "" or password == "":
        return None
    return ["GITHUB_USERNAME", "GITHUB_TOKEN"]


def _positive_limit(value: int | str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise InferenceBackendError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise InferenceBackendError(f"{name} must be a positive integer")
    return parsed


def beam_target_concurrency(
    scale_up_threshold: float | None = None,
    *,
    request_limit: int | None = None,
) -> int:
    """In-flight requests per container before Beam adds another replica.

    A caller-supplied ``request_limit`` wins. Otherwise ``BEAM_TARGET_CONCURRENCY``
    overrides the default of 16.
    """
    del scale_up_threshold
    if request_limit is not None:
        return _positive_limit(request_limit, "request_limit")
    from provider_service.limits import DEFAULT_REQUEST_LIMIT

    raw = _first_nonempty(os.getenv("BEAM_TARGET_CONCURRENCY"))
    if raw == "":
        return DEFAULT_REQUEST_LIMIT
    return _positive_limit(raw, "BEAM_TARGET_CONCURRENCY")


def runpod_target_concurrency(
    scale_up_threshold: float | None = None,
    *,
    request_limit: int | None = None,
) -> int:
    """In-flight requests per worker before Runpod adds another worker.

    A caller-supplied ``request_limit`` wins. Otherwise ``RUNPOD_TARGET_CONCURRENCY``
    overrides the default of 16.
    """
    del scale_up_threshold
    if request_limit is not None:
        return _positive_limit(request_limit, "request_limit")
    from provider_service.limits import DEFAULT_REQUEST_LIMIT

    raw = _first_nonempty(os.getenv("RUNPOD_TARGET_CONCURRENCY"))
    if raw == "":
        return DEFAULT_REQUEST_LIMIT
    return _positive_limit(raw, "RUNPOD_TARGET_CONCURRENCY")


def provider_from_url(url: str | None) -> str:
    """Infer which provider should authenticate a request from its host."""
    from urllib.parse import urlparse

    host = (urlparse(str(url or "")).hostname or "").lower()
    if host.endswith("modal.run") or ".modal." in host:
        return "modal"
    if "runpod.ai" in host or host.endswith("runpod.io"):
        return "runpod"
    if host.endswith("beam.cloud") or ".beam.cloud" in host:
        return "beam"
    return "inferx"


def registry_credentials() -> tuple[str, str]:
    username = _first_nonempty(os.getenv("GHCR_INFERENCE_IMAGE_USERNAME"))
    password = _first_nonempty(os.getenv("GHCR_INFERENCE_IMAGE_READ_TOKEN"))
    return username, password
