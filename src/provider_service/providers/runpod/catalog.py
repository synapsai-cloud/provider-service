"""Live Runpod serverless GPU stock.

https://docs.runpod.io/api-reference-v2/catalog/list-gpu-types
Availability is requested with ``include=AVAILABILITY`` and ``product=SERVERLESS``.
Workers are placed in US secure-cloud data centers, so the read uses that
same cloud and country. A short cache keeps one deploy from asking twice.
"""

from __future__ import annotations

import time

import httpx

from provider_service.providers.runpod.gpus import pools_with_stock, stocked_pool_ids

_TTL_S = 30.0
_cache: dict[int, tuple[float, tuple]] = {}


def clear_stock_cache() -> None:
    _cache.clear()


def pools_in_stock(gpu_count: int = 1) -> tuple:
    """``RUNPOD_GPU_POOLS`` rows with US secure-cloud serverless capacity for ``gpu_count``."""
    count = max(1, int(gpu_count or 1))
    now = time.monotonic()
    hit = _cache.get(count)
    if hit is not None and now - hit[0] < _TTL_S:
        return hit[1]
    pools = pools_with_stock(stocked_pool_ids(_catalog_gpus(count)))
    _cache[count] = (now, pools)
    return pools


def _catalog_gpus(gpu_count: int) -> list:
    from provider_service.settings import InferenceBackendError, runpod_api_base, runpod_api_key

    key = runpod_api_key()
    if key == "":
        raise InferenceBackendError("RUNPOD_API_KEY is required to read Runpod GPU capacity")
    url = f"{runpod_api_base()}/v2/catalog/gpus"
    try:
        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {key}"},
            params={
                "include": "AVAILABILITY",
                "product": "SERVERLESS",
                "count": gpu_count,
                "cloud": "SECURE",
                "countryCodes": "US",
            },
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise InferenceBackendError(f"Runpod GPU catalog request failed: {exc}") from exc
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}
    if response.status_code >= 400:
        detail = ""
        if isinstance(payload, dict):
            detail = str(payload.get("detail") or payload.get("title") or payload)
        raise InferenceBackendError(
            f"Runpod GET /v2/catalog/gpus failed ({response.status_code}): {detail}"
        )
    if isinstance(payload, list):
        return payload
    gpus = payload.get("gpus") if isinstance(payload, dict) else None
    if not isinstance(gpus, list):
        raise InferenceBackendError("Runpod GPU catalog did not include gpus")
    return gpus
