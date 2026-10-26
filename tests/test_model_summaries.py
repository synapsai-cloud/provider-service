"""Instance status comes back as one ModelStatusSummary per model."""

from provider_service.providers.inferx.app import status_for_pods
from provider_service.providers.schema import ModelStatusSummary, collect_model_summaries


def test_snapshot_loading_pod_stays_sleeping():
    """A Snapshot pod in Loading is creating a snapshot, not a deployment."""
    pod = {
        "objType": "pod",
        "name": "tn/default/qwen/1372747/1372751",
        "object": {
            "spec": {"funcname": "qwen", "create_type": "Snapshot"},
            "status": {"state": "Loading"},
        },
    }
    assert status_for_pods([pod]) == ("sleeping", 0)
    assert status_for_pods([
        {"id": "pod-1", "state": "Loading", "create_type": "Snapshot"},
    ]) == ("sleeping", 0)
    assert status_for_pods([
        {"id": "pod-1", "state": "Creating", "create_type": "Normal"},
    ]) == ("building", 0)


def test_collect_returns_one_model_status_summary():
    class Backend:
        backend_name = "inferx"

        def summarize_instances(self, pods, *, min_instances=0):
            from provider_service.providers.inferx.app import summarize_pod_instances

            return summarize_pod_instances(pods, min_instances=min_instances)

    rows = [
        {"id": "pod-1", "name": "pod-1", "funcname": "m1", "state": "Running"},
        {"id": "pod-2", "name": "pod-2", "funcname": "m-build", "state": "Pending"},
    ]
    result = collect_model_summaries(
        Backend(),
        rows,
        [{"key": "m1", "aliases": ["m1"], "min_instances": 0}],
    )
    by_key = {item["key"]: item for item in result}
    ready = by_key["m1"]["summary"]
    assert isinstance(ready, ModelStatusSummary)
    assert ready.status == "ready"
    assert ready.active_instances == 1
    assert ready.serving_instances[0].id == "pod-1"
    building = by_key["m-build"]["summary"]
    assert building.status == "building"
    assert building.active_instances == 0
    assert by_key["m-build"]["matched"] is False
