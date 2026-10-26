"""GET /health offers the corresponding source to network users."""

from provider_service.server import DEFAULT_SOURCE_URL, health


def test_health_offers_the_published_source(monkeypatch):
    monkeypatch.delenv("PROVIDER_SERVICE_SOURCE_URL", raising=False)
    body = health()
    assert body["ok"] is True
    assert body["license"] == "AGPL-3.0-only"
    assert body["source"] == DEFAULT_SOURCE_URL
    assert body["source"] == "https://github.com/synapsai-cloud/provider-service"


def test_health_offers_a_modified_source(monkeypatch):
    monkeypatch.setenv("PROVIDER_SERVICE_SOURCE_URL", "https://example.com/provider-service")
    assert health()["source"] == "https://example.com/provider-service"
