import asyncio

from provider_service.providers.beam.adapter import BeamAdapter
from provider_service.settings import (
    InferenceBackendError,
    beam_checkpoint_enabled,
    beam_gpu,
    beam_ondemand_gpu,
    beam_ondemand_only,
    beam_pod_memory,
    resolve_inference_backend,
)


def test_backend_must_be_set(monkeypatch):
    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
    try:
        resolve_inference_backend()
        raise AssertionError("expected InferenceBackendError")
    except InferenceBackendError as exc:
        assert "must be set" in str(exc)
    assert resolve_inference_backend("modal") == "modal"
    monkeypatch.setenv("INFERENCE_BACKEND", "beam")
    assert resolve_inference_backend() == "beam"


def test_backend_rejects_unknown_name():
    try:
        resolve_inference_backend("not-a-backend")
        raise AssertionError("expected InferenceBackendError")
    except Exception as exc:
        assert "INFERENCE_BACKEND" in str(exc)


def test_beam_gpu_picks_smallest_fit():
    assert beam_gpu(16000, 1) == "RTX4090"
    assert beam_gpu(24000, 1) == "RTX4090"
    assert beam_gpu(24000, 2) == "RTX4090"
    assert beam_gpu(32 * 1024, 1) == "RTX5090"
    assert beam_gpu(80000, 1) == "H100"
    try:
        beam_gpu(96 * 1024, 1)
        raise AssertionError("expected InferenceBackendError")
    except InferenceBackendError as exc:
        assert "80 GB" in str(exc)


def test_beam_ondemand_gpu_uses_the_wider_catalog():
    assert beam_ondemand_gpu(16000, 1) == "RTX4090"
    assert beam_ondemand_gpu(40 * 1024, 1) == "A6000"
    assert beam_ondemand_gpu(48 * 1024, 1) == "A6000"
    assert beam_ondemand_gpu(80 * 1024, 1) == "A100-80"
    assert beam_ondemand_gpu(96 * 1024, 1) == "RTXPro6000"
    assert beam_ondemand_gpu(141 * 1024, 1) == "H200"
    assert beam_ondemand_gpu(180 * 1024, 1) == "B200"
    try:
        beam_ondemand_gpu(200 * 1024, 1)
        raise AssertionError("expected InferenceBackendError")
    except InferenceBackendError as exc:
        assert "180 GB" in str(exc)


def test_beam_pod_memory_matches_vram(monkeypatch):
    monkeypatch.delenv("BEAM_POD_MEMORY", raising=False)
    assert beam_pod_memory(12 * 1024, 1) == "16Gi"
    assert beam_pod_memory(16 * 1024, 1) == "16Gi"
    assert beam_pod_memory(24 * 1024, 1) == "24Gi"
    assert beam_pod_memory(80 * 1024, 1) == "80Gi"
    assert beam_pod_memory(24 * 1024, 2) == "48Gi"
    monkeypatch.setenv("BEAM_POD_MEMORY", "16Gi")
    assert beam_pod_memory(80 * 1024, 1) == "16Gi"


def test_beam_ondemand_only_follows_the_env(monkeypatch):
    monkeypatch.setenv("BEAM_ONDEMAND_ONLY", "0")
    assert beam_ondemand_only() is False
    monkeypatch.setenv("BEAM_ONDEMAND_ONLY", "1")
    assert beam_ondemand_only() is True


def test_beam_checkpoint_enabled_by_default(monkeypatch):
    monkeypatch.delenv("BEAM_CHECKPOINT_ENABLED", raising=False)
    assert beam_checkpoint_enabled() is True
    monkeypatch.setenv("BEAM_CHECKPOINT_ENABLED", "0")
    assert beam_checkpoint_enabled() is False


def test_beam_adapter_starts_the_callers_command(monkeypatch):
    captured = {}
    command = ["python", "server.py", "--task", "text-generation"]

    def _deploy(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "url": "https://synapsai-m1-8000.app.beam.cloud",
            "image": kwargs["image"],
            "gpu": "L4",
        }

    monkeypatch.setattr(
        "provider_service.providers.beam.app.deploy_inference_pod",
        _deploy,
    )
    adapter = BeamAdapter()

    async def _run():
        result = await adapter.deploy_model(
            model_id="m1",
            command=command,
            image="ghcr.io/example/inference:1",
            vram_mb=24000,
        )
        assert result["url"] == "https://synapsai-m1-8000.app.beam.cloud"
        assert adapter.model_inference_url("m1") == "https://synapsai-m1-8000.app.beam.cloud"

    asyncio.run(_run())
    assert captured["image"] == "ghcr.io/example/inference:1"
    assert captured["command"] == command


def test_custom_image_deploy_sets_checkpoint_readiness(monkeypatch):
    captured = {}

    def _deploy(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "url": "https://synapsai-m1-8000.app.beam.cloud",
            "persistent_url": "",
            "burst_url": "https://synapsai-m1-8000.app.beam.cloud",
            "persistent_instances": 0,
            "tasks_per_container": 1,
        }

    monkeypatch.setattr(
        "provider_service.providers.beam.app.deploy_inference_pod",
        _deploy,
    )
    adapter = BeamAdapter()

    async def _run():
        await adapter.deploy_custom_function(
            model_id="m1",
            image="ghcr.io/org/app:1.0",
            commands=["python", "server.py"],
        )

    asyncio.run(_run())
    assert captured["untrusted"] is True
    assert captured["readiness_path"] == "/system/ready"
    assert captured["command"] == ["python", "server.py"]
