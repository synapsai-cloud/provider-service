"""GPU inference provider adapters (InferX, Modal, Runpod, Beam)."""

from __future__ import annotations


def build_inference_backend(name: str | None = None):
    """Construct one adapter. ``name`` overrides ``INFERENCE_BACKEND``."""
    from provider_service.settings import resolve_inference_backend

    resolved = resolve_inference_backend(name)
    if resolved == "modal":
        from provider_service.providers.modal.adapter import ModalAdapter

        return ModalAdapter()
    if resolved == "runpod":
        from provider_service.providers.runpod.adapter import RunpodAdapter

        return RunpodAdapter()
    if resolved == "beam":
        from provider_service.providers.beam.adapter import BeamAdapter

        return BeamAdapter()
    from provider_service.providers.inferx.adapter import InferxAdapter

    return InferxAdapter()


def build_inference_backends() -> dict:
    """Adapters for the fallback backend and every provider named in the routes."""
    from provider_service.settings import backend_names_in_use

    return {name: build_inference_backend(name) for name in backend_names_in_use()}
