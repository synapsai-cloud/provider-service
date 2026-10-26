"""Modal GPU cards and list prices.

https://modal.com/docs/reference/modal.gpu
https://modal.com/pricing — GPU tasks only. CPU and RAM are not included.
The third field is USD/hr (published $/sec × 3600).
"""

MODAL_GPUS = (
    (16 * 1024, "T4", 0.5904),          # $0.000164/sec
    (24 * 1024, "L4", 0.7992),          # $0.000222/sec
    (40 * 1024, "A100-40GB", 2.0988),   # $0.000583/sec
    (48 * 1024, "L40S", 1.9512),        # $0.000542/sec
    (80 * 1024, "A100-80GB", 2.4984),   # $0.000694/sec
    (141 * 1024, "H200", 4.5396),       # $0.001261/sec
)
