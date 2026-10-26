import asyncio
from unittest.mock import AsyncMock

from provider_service.providers.beam.adapter import BeamAdapter
from provider_service.providers.inferx.adapter import InferxAdapter
from provider_service.providers.modal.adapter import ModalAdapter
from provider_service.settings import (
    InferenceBackendError,
    beam_checkpoint_enabled,
    beam_gpu,
    beam_ondemand_gpu,
    beam_ondemand_only,
    beam_pod_memory,
    modal_app_name,
    modal_gpu,
    modal_proxy_bearer,
    resolve_inference_backend,
)


def test_backend_defaults_to_inferx(monkeypatch):
    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
    assert resolve_inference_backend() == "inferx"
    assert resolve_inference_backend("modal") == "modal"


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


def test_modal_gpu_picks_smallest_fit():
    assert modal_gpu(16000, 1) == "T4"
    assert modal_gpu(24000, 1) == "L4"
    assert modal_gpu(24000, 2) == "L4:2"
    assert modal_gpu(80000, 1) == "A100-80GB"


def test_modal_gpu_env_override(monkeypatch):
    monkeypatch.setenv("MODAL_GPU", "H100")
    assert modal_gpu(8000, 1) == "H100"
    assert modal_gpu(8000, 4) == "H100:4"


def test_modal_app_name_is_stable():
    assert modal_app_name("MyModel") == "synapsai-mymodel"


def test_proxy_bearer_joins_token_pair(monkeypatch):
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-1")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-2")
    assert modal_proxy_bearer() == "wk-1.ws-2"


def test_modal_adapter_starts_the_callers_command(monkeypatch):
    captured = {}
    command = ["python", "server.py", "--task", "text-generation"]

    def _deploy(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "url": "https://example.modal.run", "image": kwargs["image"], "gpu": "L4"}

    monkeypatch.setattr(
        "provider_service.providers.modal.app.deploy_inference_server",
        _deploy,
    )
    adapter = ModalAdapter()

    async def _run():
        result = await adapter.deploy_model(
            model_id="m1",
            command=command,
            image="ghcr.io/example/inference:1",
            vram_mb=24000,
            min_instances=0,
            max_instances=2,
            worker_timeout=120,
            workdir="/opt/app",
            prepare_python=True,
        )
        assert result["url"] == "https://example.modal.run"
        assert adapter.model_inference_url("m1") == "https://example.modal.run"
        assert adapter.openai_api_base("m1") == "https://example.modal.run/v1"

    asyncio.run(_run())
    assert captured["image"] == "ghcr.io/example/inference:1"
    assert captured["prepare_python"] is True
    assert captured["command"] == command
    assert captured["workdir"] == "/opt/app"
    assert captured["min_instances"] == 0
    assert captured["max_instances"] == 2


def test_inferx_deploy_model_forwards_the_task():
    adapter = InferxAdapter.__new__(InferxAdapter)
    adapter.deploy_vllm_function = AsyncMock(return_value={"ok": True})

    async def _run():
        await adapter.deploy_model(model_id="m1", hf_model="org/model", task="text-generation", vram_mb=24000)

    asyncio.run(_run())
    adapter.deploy_vllm_function.assert_awaited_once()
    assert adapter.deploy_vllm_function.await_args.kwargs["task"] == "text-generation"
    assert adapter.deploy_vllm_function.await_args.kwargs["hf_model"] == "org/model"


def test_build_inference_app_registers_a_server():
    from provider_service.providers.modal.app import build_inference_app

    app = build_inference_app(
        app_name="synapsai-m1",
        command=["python", "server.py"],
        image="ghcr.io/example/inference:1",
        gpu="L4",
        min_containers=0,
        max_containers=1,
        scaledown_window=300,
        hf_token=None,
    )
    assert app.name == "synapsai-m1"


def test_managed_image_exposes_python_without_add_python(monkeypatch):
    import modal

    from provider_service.providers.modal.app import _inference_image

    captured = {}

    def from_registry(tag, secret=None, **kwargs):
        captured["tag"] = tag
        captured["kwargs"] = kwargs
        return "image"

    monkeypatch.setattr(modal.Image, "from_registry", from_registry)
    assert _inference_image("ghcr.io/example/inference:1", prepare_python=True) == "image"
    assert "add_python" not in captured["kwargs"]
    commands = captured["kwargs"]["setup_dockerfile_commands"]
    assert "ENTRYPOINT []" in commands
    assert any("ln -sfn" in command and "/usr/local/bin/python" in command for command in commands)
    assert "secret" not in captured["kwargs"]


def test_custom_image_keeps_its_own_python(monkeypatch):
    import modal

    from provider_service.providers.modal.app import _inference_image

    captured = {}

    def from_registry(tag, secret=None, **kwargs):
        captured["kwargs"] = kwargs
        return "image"

    monkeypatch.setattr(modal.Image, "from_registry", from_registry)
    assert _inference_image("ghcr.io/example/custom:1") == "image"
    assert "add_python" not in captured["kwargs"]
    assert "setup_dockerfile_commands" not in captured["kwargs"]
