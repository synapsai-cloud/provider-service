"""Beam inference provider adapter."""

from __future__ import annotations


def build_inference_backend(name: str | None = None):
    """Construct the Beam adapter."""
    from provider_service.providers.beam.adapter import BeamAdapter

    return BeamAdapter()


def build_inference_backends() -> dict:
    return {"beam": build_inference_backend()}
