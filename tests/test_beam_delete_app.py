"""Deleting a Beam deployment stops it, drops its 9P volume, then the dashboard app."""

from types import SimpleNamespace

import pytest

from provider_service.providers.beam import app as beam_app


def _ok():
    return SimpleNamespace(ok=True, err_msg="")


def test_delete_named_deployment_stops_then_deletes_volume_and_app(monkeypatch):
    deployment = SimpleNamespace(id="dep-1", app_id="app-1", name="synapsai-qwen", stub_id="stub-1")
    container = SimpleNamespace(deployment_id="dep-1", stub_id="stub-1", container_id="pod-1")
    order: list[tuple] = []

    gateway = SimpleNamespace(
        stop_deployment=lambda request: order.append(("stop", request.id)) or _ok(),
        list_containers=lambda request: order.append(("containers",)) or SimpleNamespace(ok=True, err_msg="", containers=[container]),
        stop_container=lambda request: order.append(("stop-container", request.container_id)) or _ok(),
        delete_deployment=lambda request: order.append(("delete", request.id)) or _ok(),
    )
    deleted_apps: list[str] = []
    deleted_volumes: list[str] = []

    monkeypatch.setattr(beam_app, "list_deployments_named", lambda name: [deployment])
    monkeypatch.setattr(beam_app, "_gateway", lambda: gateway)
    monkeypatch.setattr(beam_app, "_app_ids_named", lambda name: ["app-1", "app-orphan"])
    monkeypatch.setattr(
        beam_app,
        "_delete_beam_app",
        lambda app_id: deleted_apps.append(app_id) or order.append(("app", app_id)),
    )
    monkeypatch.setattr(
        "provider_service.providers.beam.capacity.release_app_volumes",
        lambda name: deleted_volumes.append(name) or order.append(("volume", name)),
    )

    beam_app._delete_named_deployment("synapsai-qwen")

    assert order == [
        ("stop", "dep-1"),
        ("containers",),
        ("stop-container", "pod-1"),
        ("delete", "dep-1"),
        ("volume", "synapsai-qwen"),
        ("app", "app-1"),
        ("app", "app-orphan"),
    ]
    assert deleted_apps == ["app-1", "app-orphan"]
    assert deleted_volumes == ["synapsai-qwen"]


def test_delete_named_deployment_continues_after_stop_fails(monkeypatch):
    deployment = SimpleNamespace(id="dep-1", app_id="app-1", name="synapsai-qwen", stub_id="stub-1")
    order: list[str] = []

    def stop_deployment(request):
        order.append("stop")
        raise beam_app.BeamBackendError("stop rejected")

    gateway = SimpleNamespace(
        stop_deployment=stop_deployment,
        list_containers=lambda request: order.append("containers") or SimpleNamespace(ok=True, err_msg="", containers=[]),
        delete_deployment=lambda request: order.append("delete") or _ok(),
    )
    monkeypatch.setattr(beam_app, "list_deployments_named", lambda name: [deployment])
    monkeypatch.setattr(beam_app, "_gateway", lambda: gateway)
    monkeypatch.setattr(beam_app, "_app_ids_named", lambda name: ["app-1"])
    monkeypatch.setattr(beam_app, "_delete_beam_app", lambda app_id: order.append("app"))
    monkeypatch.setattr(
        "provider_service.providers.beam.capacity.release_app_volumes",
        lambda name: order.append("volume"),
    )

    with pytest.raises(beam_app.BeamBackendError, match="stop rejected"):
        beam_app._delete_named_deployment("synapsai-qwen")

    assert order == ["stop", "containers", "delete", "volume", "app"]


def test_stop_inference_pod_releases_the_pool_after_a_burst_failure(monkeypatch):
    calls: list[str] = []

    def delete_named(name: str) -> None:
        calls.append(name)
        if not name.endswith("-warm"):
            raise beam_app.BeamBackendError("burst failed")

    released: list[str] = []
    monkeypatch.setattr(beam_app, "_delete_named_deployment", delete_named)
    monkeypatch.setattr(
        "provider_service.providers.beam.capacity.release_ondemand_pool",
        lambda model_id: released.append(model_id),
    )

    with pytest.raises(beam_app.BeamBackendError, match="burst failed"):
        beam_app.stop_inference_pod("qwen")

    assert any(name.endswith("-warm") for name in calls)
    assert released == ["qwen"]


def test_app_rows_reads_paginated_data():
    rows, cursor = beam_app._app_rows({"data": [{"name": "synapsai-qwen", "id": "app-1"}], "next": "page-2"})
    assert rows[0]["id"] == "app-1"
    assert cursor == "page-2"
