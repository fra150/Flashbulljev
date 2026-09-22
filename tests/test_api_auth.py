"""Auth + rate-limit tests for the API."""
from fastapi.testclient import TestClient

import src.flashbulljev.api as api_mod


def _client():
    return TestClient(api_mod.app)


def test_no_key_allows_by_default(monkeypatch):
    monkeypatch.delenv("FLASHBULLJEV_API_KEY", raising=False)
    monkeypatch.setenv("FLASHBULLJEV_RPS", "0")
    r = _client().post("/v1/decisions", json={"state": "x", "questions": {}})
    assert r.status_code == 200


def test_wrong_key_401(monkeypatch):
    monkeypatch.setenv("FLASHBULLJEV_API_KEY", "secret123")
    monkeypatch.setenv("FLASHBULLJEV_RPS", "0")
    c = _client()
    assert c.post("/v1/decisions", json={"state": "x", "questions": {}}).status_code == 401
    assert c.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 401
    ok = c.post("/v1/decisions", json={"state": "x", "questions": {}},
                headers={"Authorization": "Bearer secret123"})
    assert ok.status_code == 200
    assert c.get("/health").status_code == 200  # health stays public


def test_rate_limit_429(monkeypatch):
    monkeypatch.delenv("FLASHBULLJEV_API_KEY", raising=False)
    monkeypatch.setenv("FLASHBULLJEV_RPS", "1")
    api_mod._rate_hits.clear()
    c = _client()
    assert c.post("/v1/decisions", json={"state": "x", "questions": {}}).status_code == 200
    assert c.post("/v1/decisions", json={"state": "x", "questions": {}}).status_code == 429
    monkeypatch.setenv("FLASHBULLJEV_RPS", "0")
