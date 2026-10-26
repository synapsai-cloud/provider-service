import asyncio

import pytest

from provider_service.providers.beam import app as beam_app
from provider_service.providers.beam.app import BeamBackendError
from provider_service.providers.beam.adapter import BeamAdapter
from provider_service.providers.inferx.adapter import InferxAdapter
from provider_service.providers.modal import app as modal_app
from provider_service.providers.modal.adapter import ModalAdapter
from provider_service.providers.runpod import app as runpod_app
from provider_service.providers.runpod.adapter import RunpodAdapter
from provider_service.providers.schema import tail_log_text


class _LogEntry:
    def __init__(self, message: str):
        self.message = message


class _ServerLogs:
    def tail(self, entries: int):
        yield _LogEntry("loading Qwen/Qwen3.5-4B\n")
        yield _LogEntry("ValueError: CUDA_VISIBLE_DEVICES environment variable is not set\n")
        assert entries >= 1


def test_tail_log_text_keeps_the_last_lines():
    assert tail_log_text("a\nb\nc", 2) == "b\nc"


def test_inferx_get_model_logs_reads_matching_pods():
    class Backend:
        def list_instances(self):
            return [
                {
                    "funcname": "other-model",
                    "fprevision": "1",
                    "id": "skip",
                    "state": "Running",
                    "tenant": "tn",
                    "namespace": "default",
                },
                {
                    "funcname": "text-generation-abc",
                    "fprevision": "4",
                    "id": "pod-9",
                    "state": "Failed",
                    "tenant": "tn",
                    "namespace": "default",
                },
            ]

        def read_pod_log(self, **kwargs):
            assert kwargs["funcname"] == "text-generation-abc"
            assert kwargs["fprevision"] == "4"
            assert kwargs["pod_id"] == "pod-9"
            return "one\ntwo\nCUDA_VISIBLE_DEVICES environment variable is not set"

    adapter = InferxAdapter(backend=Backend())

    async def _run():
        return await adapter.get_model_logs("text-generation-abc", tail=2)

    text = asyncio.run(_run())
    assert "pod-9 (Failed)" in text
    assert text.endswith("two\nCUDA_VISIBLE_DEVICES environment variable is not set")
    assert "one" not in text.split("-----\n", 1)[1]


def test_modal_get_model_logs(monkeypatch):
    monkeypatch.setattr(modal_app, "_server", lambda app_name: type("Server", (), {"logs": _ServerLogs()})())
    adapter = ModalAdapter()

    async def _run():
        return await adapter.get_model_logs("text-generation-abc", tail=20)

    text = asyncio.run(_run())
    assert "CUDA_VISIBLE_DEVICES environment variable is not set" in text


def test_runpod_sse_data_line_reads_container_line():
    raw = 'data: {"source": "container", "line": "Error loading model", "ts": "2026-10-16T14:16:04Z"}'
    assert runpod_app._sse_data_line(raw) == "Error loading model"
    assert runpod_app._sse_data_line("id: 1") is None


def test_runpod_get_model_logs(monkeypatch):
    monkeypatch.setattr(
        runpod_app,
        "find_endpoint",
        lambda model_id: {"id": "ep-1", "name": "synapsai-text-generation-abc"},
    )
    monkeypatch.setattr(
        runpod_app,
        "_list_workers",
        lambda endpoint_id: [{"id": "worker-1", "status": "RUNNING"}],
    )

    class Response:
        status_code = 200

        def iter_lines(self):
            yield 'data: {"line": "Error loading model Qwen/Qwen3.5-4B"}'
            yield 'data: {"line": "CUDA_VISIBLE_DEVICES environment variable is not set"}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(runpod_app, "_api_key", lambda: "test-key")
    monkeypatch.setattr(runpod_app.httpx, "stream", lambda *args, **kwargs: Response())
    adapter = RunpodAdapter()

    async def _run():
        return await adapter.get_model_logs("text-generation-abc")

    text = asyncio.run(_run())
    assert "worker-1 (Running)" in text
    assert "CUDA_VISIBLE_DEVICES environment variable is not set" in text


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
