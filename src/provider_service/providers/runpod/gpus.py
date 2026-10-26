"""Runpod serverless GPU pools and flex-worker list prices.

https://docs.runpod.io/references/gpu-types
https://www.runpod.io/pricing (serverless, October 2026).
The third field is USD/hr. AMPERE_48 is a 48 GB pool; 40 GB is only the
selection ceiling before ADA_48_PRO.
"""

RUNPOD_GPU_POOLS = (
    (16 * 1024, "AMPERE_16", 0.58),
    (24 * 1024, "AMPERE_24", 0.69),
    (40 * 1024, "AMPERE_48", 1.22),
    (48 * 1024, "ADA_48_PRO", 1.75),
    (80 * 1024, "AMPERE_80", 2.72),
    (141 * 1024, "HOPPER_141", 5.93),
)
