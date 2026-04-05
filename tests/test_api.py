from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("EDGE_FORCE_PYTHON_ENGINE", "1")
os.environ.setdefault("EDGE_SIMULATE_CAMERAS", "1")

from api.main import app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "engine" in body


def test_infer(client: TestClient) -> None:
    tensor = [0.1] * (3 * 224 * 224)
    r = client.post("/infer", json={"tensor": tensor, "channels": 3, "height": 224, "width": 224})
    assert r.status_code == 200
    body = r.json()
    assert len(body["output"]) == 64
    assert body["latency_ms"] >= 0


def test_batch_infer(client: TestClient) -> None:
    t = [0.05] * (3 * 224 * 224)
    r = client.post("/batch_infer", json={"tensors": [t, t], "channels": 3, "height": 224, "width": 224})
    assert r.status_code == 200
    body = r.json()
    assert len(body["outputs"]) == 2


def test_infer_latest(client: TestClient) -> None:
    r = client.post("/infer_latest/csi-0")
    assert r.status_code == 200
    body = r.json()
    assert len(body["output"]) == 64
    assert body["latency_ms"] >= 0
