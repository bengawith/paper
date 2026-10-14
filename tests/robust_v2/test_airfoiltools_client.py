import httpx

from robust_airfoil.sources.airfoiltools import AirfoilToolsClient


def test_contact_gate_blocks_all_network(monkeypatch):
    monkeypatch.delenv("AIRFOILTOOLS_CONTACT", raising=False)
    client = AirfoilToolsClient("https://example/{polar_key}", "https://example/robots", "canary", minimum_delay_seconds=0)
    assert client.probe_canary().status == "skipped"
    assert client.fetch_many(["a", "b"], 2).stopped_early


def test_valid_body_is_accepted(monkeypatch, tmp_path):
    monkeypatch.setenv("AIRFOILTOOLS_CONTACT", "researcher@example.com")
    response = httpx.Response(200, content=b"Alpha CL CD CDp CM Top Bot\n0 0 0.01 0.008 0 1 1")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: response)
    client = AirfoilToolsClient("https://example/{polar_key}", "https://example/robots", "canary", minimum_delay_seconds=0, output_dir=tmp_path)
    result = client.fetch_polar("case")
    assert result.status == "ok"
    assert (tmp_path / "case.csv").exists()
