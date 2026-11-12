import asyncio

import pytest

from provider_service.providers.beam import app as beam_app
from provider_service.providers.beam.app import BeamBackendError
from provider_service.providers.beam.adapter import BeamAdapter
from provider_service.providers.schema import tail_log_text


def test_tail_log_text_keeps_the_last_lines():
    assert tail_log_text("a\nb\nc", 2) == "b\nc"


def test_beam_get_model_logs(monkeypatch):
    monkeypatch.setattr(
        beam_app,
        "list_pod_instances",
        lambda: [{"id": "container-1", "funcname": "text-generation-abc", "state": "Running"}],
    )

    class Context:
        token = "beam-token"
        http_url = "https://app.beam.cloud"

    class Workspace:
        workspace_id = "ws-1"

    class Gateway:
        def export_workspace_config(self, request):
            del request
            return Workspace()

    captured = {}

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {
                "logs": [
                    {"message": "Error loading model Qwen/Qwen3.5-4B from CLI"},
                    {"message": "ValueError: CUDA_VISIBLE_DEVICES environment variable is not set"},
                ]
            }

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs["params"]
        return Response()

    monkeypatch.setattr(beam_app, "_config_context", lambda: Context())
    monkeypatch.setattr(beam_app, "_gateway", lambda: Gateway())
    monkeypatch.setattr("httpx.get", fake_get)
    adapter = BeamAdapter()

    async def _run():
        return await adapter.get_model_logs("text-generation-abc", tail=50)

    text = asyncio.run(_run())
    assert captured["url"] == "https://app.beam.cloud/api/v1/logs/ws-1"
    assert captured["params"]["object_type"] == "BETA9_CONTAINER"
    assert captured["params"]["object_id"] == "container-1"
    assert "CUDA_VISIBLE_DEVICES environment variable is not set" in text


def test_beam_application_logs_keep_stdout_and_advance_the_cursor(monkeypatch):
    monkeypatch.setattr(
        beam_app,
        "list_pod_instances",
        lambda: [{"id": "container-1", "funcname": "text-generation-abc", "state": "Running"}],
    )

    class Context:
        token = "beam-token"
        http_url = "https://app.beam.cloud"

    class Workspace:
        workspace_id = "ws-1"

    class Gateway:
        def export_workspace_config(self, request):
            del request
            return Workspace()

    payloads = [
        [
            {"message": "booting", "stream": "stdout", "seq_num": 1},
            {"message": "scheduler assigned worker", "stream": "system", "seq_num": 2},
            {"message": "ready", "source": "stdout", "seq_num": 3},
            {"message": "checkpoint started", "source": "system", "seq_num": 4},
            {"message": "stderr noise", "stream": "stderr", "seq_num": 5},
        ]
    ]

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"logs": payloads[0]}

    seen_params = []

    def fake_get(url, **kwargs):
        del url
        seen_params.append(dict(kwargs["params"]))
        return Response()

    monkeypatch.setattr(beam_app, "_config_context", lambda: Context())
    monkeypatch.setattr(beam_app, "_gateway", lambda: Gateway())
    monkeypatch.setattr("httpx.get", fake_get)
    adapter = BeamAdapter()

    async def _run():
        first = await adapter.read_application_logs("text-generation-abc")
        second = await adapter.read_application_logs("text-generation-abc", cursor=first.cursor)
        payloads[0] = payloads[0] + [{"message": "request ok", "stream": "stdout", "seq_num": 6}]
        third = await adapter.read_application_logs("text-generation-abc", cursor=second.cursor)
        return first, second, third

    first, second, third = asyncio.run(_run())
    assert first.lines == ("booting", "ready")
    assert second.lines == ()
    assert third.lines == ("request ok",)
    assert "seq_num" not in seen_params[0]
    assert seen_params[1]["seq_num"] == 6
    assert seen_params[2]["seq_num"] == 6


def test_wait_for_checkpoint_returns_when_the_snapshot_is_created(monkeypatch):
    seen = {"n": 0}

    def logs(model_id, tail=200):
        assert model_id == "m1"
        assert tail == 1000
        seen["n"] += 1
        if seen["n"] == 1:
            return "container starting"
        return "Checkpoint created successfully"

    monkeypatch.setattr(beam_app, "beam_checkpoint_enabled", lambda: True)
    monkeypatch.setattr(beam_app, "get_model_logs", logs)
    monkeypatch.setattr(beam_app.time, "sleep", lambda _seconds: None)

    beam_app.wait_for_checkpoint("m1", timeout_s=30, poll_s=0)
    assert seen["n"] == 2


def test_wait_for_checkpoint_raises_when_the_archive_fails(monkeypatch):
    monkeypatch.setattr(beam_app, "beam_checkpoint_enabled", lambda: True)
    monkeypatch.setattr(
        beam_app,
        "get_model_logs",
        lambda model_id, tail=200: "Failed to copy checkpoint filesystem state: tar: file changed as we read it",
    )
    monkeypatch.setattr(beam_app.time, "sleep", lambda _seconds: None)

    with pytest.raises(BeamBackendError):
        beam_app.wait_for_checkpoint("m1", timeout_s=30, poll_s=0)


def test_wait_for_checkpoint_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(beam_app, "beam_checkpoint_enabled", lambda: False)

    def logs(*_args, **_kwargs):
        raise AssertionError("logs should not be read")

    monkeypatch.setattr(beam_app, "get_model_logs", logs)
    beam_app.wait_for_checkpoint("m1")
