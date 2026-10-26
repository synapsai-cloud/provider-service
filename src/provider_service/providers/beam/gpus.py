"""Beam GPU cards.

Serverless Pods: https://docs.beam.cloud/v2/environment/gpu
On-demand machines and GPU-task rates: https://www.beam.cloud/pricing

Pod inventory is RTX 4090, RTX 5090, and H100. ``vram_mb`` is the memory one
GPU must provide; ``gpu_count`` is applied separately.

Pod rows are the GPU-task rate only ($/sec × 3600). A burst pod also pays
GPU-attached CPU and RAM; those rates are separate. On-demand rows are the
bundled machine from-price in USD/hr, CPU and RAM included. Same-VRAM ties
take the cheaper card.
"""

# $/sec on the GPU tasks table.
_GPU_TASK_USD_PER_SEC = {
    "RTX4090": 0.000192,
    "RTX5090": 0.000303,
    "H100": 0.000972,
}

BEAM_POD_GPUS = (
    (24 * 1024, "RTX4090", _GPU_TASK_USD_PER_SEC["RTX4090"] * 3600),
    (32 * 1024, "RTX5090", _GPU_TASK_USD_PER_SEC["RTX5090"] * 3600),
    (80 * 1024, "H100", _GPU_TASK_USD_PER_SEC["H100"] * 3600),
)

# GPU-attached rates. One physical core is two vCPUs. The published 4090
# example is $0.76 for 2 cores and $0.32 for 16 GiB.
BEAM_BURST_CPU_USD_PER_PHYSICAL_CORE_HOUR = 0.000105 * 3600
BEAM_BURST_RAM_USD_PER_GIB_HOUR = 0.32 / 16

BEAM_ONDEMAND_GPUS = (
    (24 * 1024, "RTX4090", 0.44),
    (32 * 1024, "RTX5090", 0.72),
    (48 * 1024, "A6000", 0.54),
    (48 * 1024, "L40S", 0.76),
    (80 * 1024, "A100-80", 1.36),
    (80 * 1024, "H100", 1.83),
    (96 * 1024, "RTXPro6000", 1.09),
    (141 * 1024, "H200", 2.09),
    (180 * 1024, "B200", 4.11),
)
