"""Clean pipeline + API tests."""
from fastapi.testclient import TestClient

from src.flashbulljev.api import app
from src.flashbulljev.engine import FlashBullJevEngine
from src.flashbulljev.pipeline import batch_states, run_pipeline


def test_pipeline_steps_list():
    eng = FlashBullJevEngine()
    res = run_pipeline(eng, "state demo", {"q": {"type": "noul", "instructions": "x?"}})
    assert isinstance(res.steps, list)
    assert res.steps[0] == "ingest"
    assert "emit" in res.steps


def test_batch_states():
    out = batch_states(["a", "b"])
    assert isinstance(out, list)
    assert out[0] == {"state": "a"}


def test_api_decisions():
    c = TestClient(app)
    r = c.post("/v1/decisions", json={"state": "payout failing", "questions": {"is_urgent": {"type": "noul", "instructions": "urgent?"}}})
    assert r.status_code == 200
    data = r.json()
    assert "answers" in data
    assert data["usage"]["output_tokens"] == 0


def test_api_systemone():
    c = TestClient(app)
    r = c.post("/v1/systemone", json={"state": "hello", "questions": {"q": {"type": "noul", "instructions": "hi?"}}})
    assert r.status_code == 200
    assert "answers" in r.json()


def test_api_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
