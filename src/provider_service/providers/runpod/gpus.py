"""Runpod serverless GPU pools and flex-worker list prices.

https://docs.runpod.io/references/gpu-types
https://www.runpod.io/pricing (serverless, October 2026).
The third field is USD/hr. AMPERE_48 is a 48 GB pool; 40 GB is only the
selection ceiling before ADA_48_PRO.
"""

RUNPOD_GPU_POOLS = (
    (16 * 1024, "AMPERE_16", 0.58),
    (24 * 1024, "AMPERE_24", 0.75),
    (24 * 1024, "ADA_24", 1.34),
    (32 * 1024, "ADA_32_PRO", 1.79),
    (40 * 1024, "AMPERE_48", 1.22),
    (48 * 1024, "ADA_48_PRO", 1.75),
    (80 * 1024, "AMPERE_80", 2.72),
    (80 * 1024, "ADA_80_PRO", 5.39),
    (96 * 1024, "BLACKWELL_96", 3.49),
    (141 * 1024, "HOPPER_141", 6.99),
    (180 * 1024, "BLACKWELL_180", 10.79),
)

# Catalog ``availability`` values that still have machines to place.
# NONE is out of stock. Anything else is not a level the catalog documents.
_IN_STOCK = frozenset({"LOW", "MEDIUM", "HIGH"})


def stocked_pool_ids(catalog_gpus: list) -> set[str]:
    """Pool ids that have serverless stock above NONE.

    Several GPU types can share one pool. The pool is usable when any of
    those types reports LOW, MEDIUM, or HIGH.
    """
    stocked: set[str] = set()
    for gpu in catalog_gpus or []:
        if not isinstance(gpu, dict):
            continue
        pool = str(gpu.get("pool") or "").strip()
        if pool == "":
            continue
        level = str(gpu.get("availability") or "NONE").strip().upper()
        if level in _IN_STOCK:
            stocked.add(pool)
    return stocked


def pools_with_stock(stocked: set[str]) -> tuple:
    """``RUNPOD_GPU_POOLS`` rows whose pool id is in ``stocked``, in list order."""
    return tuple(row for row in RUNPOD_GPU_POOLS if row[1] in stocked)
