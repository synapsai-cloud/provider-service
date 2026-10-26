"""Beam capacity split and request routing."""

from types import SimpleNamespace

from provider_service.providers.beam.adapter import _RouteBook
from provider_service.providers.beam.capacity import (
    burst_storage_gb,
    function_name_for_app,
    split_capacity,
    volume_names_for_app,
    warm_app_name,
)
from provider_service.providers.runpod.app import endpoint_storage_gb
from provider_service.settings import synapsai_app_name


def test_split_capacity(monkeypatch):
    monkeypatch.setenv("BEAM_ONDEMAND_ONLY", "0")
    assert split_capacity(0, 2) == (0, 2)
    assert split_capacity(1, 2) == (1, 1)
    assert split_capacity(2, 2) == (2, 0)
    assert split_capacity(3, 2) == (3, 0)


def test_split_capacity_ondemand_only_reserves_the_max(monkeypatch):
    monkeypatch.setenv("BEAM_ONDEMAND_ONLY", "1")
    assert split_capacity(0, 1) == (1, 0)
    assert split_capacity(0, 2) == (2, 0)
    assert split_capacity(1, 3) == (3, 0)


def test_volume_names_keep_the_warm_sibling_off_the_burst_delete():
    model_id = "qwen"
    burst = synapsai_app_name(model_id)
    warm = warm_app_name(model_id)
    burst_checkpoint = f"checkpoint-model-cache-{burst}-afsjcet5"
    warm_checkpoint = f"checkpoint-model-cache-{warm}-cxdr9wtz"
    volumes = [
        SimpleNamespace(name=burst),
        SimpleNamespace(name=f"{burst}-cache"),
        SimpleNamespace(name=burst_checkpoint),
        SimpleNamespace(name=warm),
        SimpleNamespace(name=warm_checkpoint),
        SimpleNamespace(name="other"),
        SimpleNamespace(name="checkpoint-model-cache-other-o6gnw1ym"),
    ]
    assert volume_names_for_app(burst, volumes) == [burst, f"{burst}-cache", burst_checkpoint]
    assert volume_names_for_app(warm, volumes) == [warm, warm_checkpoint]


def test_burst_storage_reads_volume_bytes_for_the_burst_pod():
    model_id = "qwen"
    burst = synapsai_app_name(model_id)
    warm = warm_app_name(model_id)
    volumes = [
        SimpleNamespace(name=burst, size=3 * 1024**3),
        SimpleNamespace(name=f"checkpoint-model-cache-{burst}-afsjcet5", size=2 * 1024**3),
        SimpleNamespace(name=f"checkpoint-model-cache-{warm}-cxdr9wtz", size=4 * 1024**3),
        SimpleNamespace(name=warm, size=9 * 1024**3),
        SimpleNamespace(name="other", size=1024**3),
    ]
    assert burst_storage_gb(model_id, has_burst_deployment=False, volumes=volumes) == 0
    assert burst_storage_gb(model_id, has_burst_deployment=True, volumes=volumes) == 5


def test_runpod_storage_reads_the_endpoint_disk():
    assert endpoint_storage_gb(None) == 0
    assert endpoint_storage_gb({"disk": 40}) == 40
    assert endpoint_storage_gb({"template": {"disk": 80}}) == 80


def test_warm_name_strips_back_to_the_model():
    model_id = "qwen3-small"
    warm = warm_app_name(model_id)
    assert warm.endswith("-warm")

    def _strip_prefix(name: str) -> str:
        return name.removeprefix("synapsai-")

    assert function_name_for_app(warm, _strip_prefix) == function_name_for_app(
        warm.removesuffix("-warm"),
        _strip_prefix,
    )


def test_route_book_fills_the_warm_replica_before_burst():
    book = _RouteBook(
        persistent_url="https://warm.example",
        burst_url="https://burst.example",
        persistent_slots=2,
    )
    assert book.take() == ("persistent", "https://warm.example")
    assert book.take() == ("persistent", "https://warm.example")
    assert book.take() == ("burst", "https://burst.example")
    book.release("persistent")
    assert book.take() == ("persistent", "https://warm.example")


def test_beam_list_pod_instances_standardizes_status_and_warm_flag(monkeypatch):
    from provider_service.providers.beam import app as beam_app

    deployments = [
        SimpleNamespace(id="dep-warm", stub_id="stub-warm", name="synapsai-qwen-warm"),
        SimpleNamespace(id="dep-burst", stub_id="stub-burst", name="synapsai-qwen"),
    ]
    containers = [
        SimpleNamespace(
            container_id="c-warm",
            deployment_id="dep-warm",
            stub_id="stub-warm",
            status="RUNNING",
        ),
        SimpleNamespace(
            container_id="c-burst-pending",
            deployment_id="dep-burst",
            stub_id="stub-burst",
            status="PENDING",
        ),
        SimpleNamespace(
            container_id="c-burst-stopping",
            deployment_id="dep-burst",
            stub_id="stub-burst",
            status="STOPPING",
        ),
    ]
    gateway = SimpleNamespace(
        list_containers=lambda _req: SimpleNamespace(ok=True, containers=containers),
    )
    monkeypatch.setattr(beam_app, "_gateway", lambda: gateway)
    monkeypatch.setattr(beam_app, "list_deployments", lambda: deployments)

    rows = beam_app.list_pod_instances()
    by_id = {row["id"]: row for row in rows}
    assert by_id["c-warm"]["state"] == "Running"
    assert by_id["c-warm"]["status"] == "ready"
    assert by_id["c-warm"]["warm"] is True
    assert by_id["c-warm"]["backend"] == "beam"
    assert by_id["c-warm"]["funcname"] == "qwen"

    assert by_id["c-burst-pending"]["state"] == "Restoring"
    assert by_id["c-burst-pending"]["status"] == "sleeping"
    assert by_id["c-burst-pending"]["warm"] is False

    assert by_id["c-burst-stopping"]["state"] == "Stopping"
    assert by_id["c-burst-stopping"]["status"] == "stopped"
    assert by_id["c-burst-stopping"]["warm"] is False

    summary_pending_only = beam_app.summarize_pod_instances([by_id["c-burst-pending"]])
    assert summary_pending_only.status == "sleeping"
    assert summary_pending_only.active_instances == 0
    assert summary_pending_only.serving_instances == []

    summary_all = beam_app.summarize_pod_instances(rows)
    assert summary_all.status == "ready"
    assert summary_all.active_instances == 1
    assert len(summary_all.serving_instances) == 1
    assert summary_all.serving_instances[0].id == "c-warm"
    assert summary_all.serving_instances[0].warm is True


def test_runpod_and_modal_list_instances_standardize_warm_vs_burst(monkeypatch):
    import modal
    from provider_service.providers.modal import app as modal_app
    from provider_service.providers.runpod import app as runpod_app

    monkeypatch.setattr(
        runpod_app,
        "list_endpoints",
        lambda: [{"id": "ep-1", "name": "synapsai-qwen", "workers": {"min": 1, "max": 2}}],
    )
    monkeypatch.setattr(
        runpod_app,
        "_list_workers",
        lambda _ep_id: [
            {"id": "w-1", "status": "RUNNING"},
            {"id": "w-2", "status": "IDLE"},
            {"id": "w-3", "status": "INITIALIZING"},
        ],
    )
    rp_rows = runpod_app.list_endpoint_instances()
    rp_by_id = {row["id"]: row for row in rp_rows}
    assert rp_by_id["w-1"]["status"] == "ready"
    assert rp_by_id["w-1"]["warm"] is True
    assert rp_by_id["w-2"]["status"] == "ready"
    assert rp_by_id["w-2"]["warm"] is False
    assert rp_by_id["w-3"]["status"] == "sleeping"
    assert rp_by_id["w-3"]["warm"] is False

    rp_pending_summary = runpod_app.summarize_endpoint_instances([rp_by_id["w-3"]], min_instances=0)
    assert rp_pending_summary.status == "sleeping"
    assert rp_pending_summary.active_instances == 0

    modal_app._MIN_CONTAINERS["synapsai-qwen"] = 1
    monkeypatch.setattr(
        modal.experimental,
        "list_deployed_apps",
        lambda **_kwargs: [SimpleNamespace(name="synapsai-qwen", containers=2)],
    )
    monkeypatch.setattr(
        modal_app,
        "_list_containers",
        lambda _app_name: [
            SimpleNamespace(container_id="ta-1"),
            SimpleNamespace(container_id="ta-2"),
        ],
    )
    modal_rows = modal_app.list_server_instances()
    assert len(modal_rows) == 2
    assert modal_rows[0]["id"] == "ta-1"
    assert modal_rows[0]["status"] == "ready"
    assert modal_rows[0]["warm"] is True
    assert modal_rows[1]["id"] == "ta-2"
    assert modal_rows[1]["status"] == "ready"
    assert modal_rows[1]["warm"] is False

