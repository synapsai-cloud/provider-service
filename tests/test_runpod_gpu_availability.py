"""Runpod GPU choice when ``by_availability`` is set."""

from provider_service.providers.runpod.app import _container_body
from provider_service.providers.runpod.catalog import clear_stock_cache, pools_in_stock
from provider_service.providers.runpod.gpus import pools_with_stock, stocked_pool_ids
from provider_service.settings import InferenceBackendError, runpod_gpu_pool


def test_stocked_pool_ids_keep_low_and_drop_none():
    stocked = stocked_pool_ids(
        [
            {"pool": "AMPERE_24", "availability": "NONE"},
            {"pool": "ADA_24", "availability": "LOW"},
            {"pool": "AMPERE_48", "availability": "HIGH"},
        ]
    )
    assert [row[1] for row in pools_with_stock(stocked)] == ["ADA_24", "AMPERE_48"]


def test_endpoint_body_uses_the_in_stock_pool(monkeypatch):
    monkeypatch.setattr(
        "provider_service.providers.runpod.app.us_data_center_ids",
        lambda: ["US-TX-3"],
    )
    monkeypatch.setattr(
        "provider_service.providers.runpod.catalog.pools_in_stock",
        lambda gpu_count=1: pools_with_stock({"ADA_24"}),
    )
    body = _container_body(
        command=["python", "-m", "server"],
        image="docker.io/example/inference:latest",
        gpu_count=1,
        vram_mb=20 * 1024,
        min_instances=0,
        max_instances=1,
        worker_timeout=None,
        scale_up_threshold=None,
        hf_token=None,
        by_availability=True,
    )
    assert body["gpu"] == {"pools": ["ADA_24"], "count": 1}


def test_by_availability_with_no_stock_raises(monkeypatch):
    monkeypatch.setattr("provider_service.providers.runpod.catalog.pools_in_stock", lambda gpu_count=1: ())
    try:
        runpod_gpu_pool(16 * 1024, by_availability=True)
    except InferenceBackendError as exc:
        assert "no serverless GPU pool" in str(exc)
    else:
        raise AssertionError("expected InferenceBackendError")


def test_catalog_request_is_us_secure_serverless(monkeypatch):
    clear_stock_cache()
    seen = {}

    class _Response:
        status_code = 200
        text = ""

        def json(self):
            return {"gpus": [{"pool": "BLACKWELL_96", "availability": "HIGH"}]}

    def _get(url, **kwargs):
        seen["url"] = url
        seen["params"] = kwargs["params"]
        return _Response()

    monkeypatch.setattr("provider_service.settings.runpod_api_key", lambda: "test-key")
    monkeypatch.setattr("provider_service.settings.runpod_api_base", lambda: "https://api.runpod.io")
    monkeypatch.setattr("provider_service.providers.runpod.catalog.httpx.get", _get)
    pools = pools_in_stock(2)
    assert [row[1] for row in pools] == ["BLACKWELL_96"]
    assert seen["url"] == "https://api.runpod.io/v2/catalog/gpus"
    assert seen["params"]["product"] == "SERVERLESS"
    assert seen["params"]["count"] == 2
    assert seen["params"]["cloud"] == "SECURE"
    assert seen["params"]["countryCodes"] == "US"
    clear_stock_cache()
