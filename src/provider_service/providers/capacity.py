"""Beam capacity failures.

The gateway rejects a request it can never place with HTTP 503 and
``{"error": "No compute capacity for GPU type ..."}``.
``concurrency_limit_reached`` is a workspace quota, not missing GPUs.
"""

from __future__ import annotations

_PHRASES = (
    "no compute capacity",
    "no serverless capacity",
    "no compatible capacity",
    "had no capacity",
    "insufficient capacity",
    "gpu capacity",
)


def beam_out_of_capacity(message: str) -> bool:
    text = str(message or "").lower()
    if "concurrency_limit_reached" in text or "cpu quota exceeded" in text:
        return False
    return any(phrase in text for phrase in _PHRASES)


def raise_if_beam_capacity(message: str) -> None:
    if not beam_out_of_capacity(message):
        return
    from provider_service.settings import OutOfCapacityError

    raise OutOfCapacityError(message)
