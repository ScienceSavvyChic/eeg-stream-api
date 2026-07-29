from fastapi.testclient import TestClient

from eeg_stream.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_demo_window():
    r = client.get("/v1/demo/window?channels=4&samples=256")
    body = r.json()
    assert len(body["data_uv"]) == 4
    assert len(body["data_uv"][0]) == 256


def test_features_from_demo():
    demo = client.get("/v1/demo/window").json()
    r = client.post("/v1/features", json={"sfreq_hz": demo["sfreq_hz"], "data_uv": demo["data_uv"]})
    assert r.status_code == 200
    out = r.json()
    assert "focus_index" in out
    assert len(out["band_power"]) > 0


def test_mock_pieeg():
    r = client.get("/v1/mock/pieeg/next")
    assert r.status_code == 200
    assert "sequence" in r.json()
