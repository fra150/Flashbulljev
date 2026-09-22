"""FastAPI service with a Jev-compatible interface and real/fake backends."""
from __future__ import annotations

import hmac
import os
import threading
import time
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from .backends import get_backend
from .calibration import load_temperature
from .engine import MODEL_ID, FlashBullJevEngine
from .pipeline import run_pipeline

app = FastAPI(title="flashbulljev", version="0.4.0")

_rate_lock = threading.Lock()
_rate_hits: Dict[str, List[float]] = {}


def _api_key() -> str:
    """Bearer key from env (empty = auth disabled, localhost dev)."""
    return os.getenv("FLASHBULLJEV_API_KEY", "")


def _rate_limit() -> float:
    """Max requests/sec per IP from env (0 = off)."""
    try:
        return max(0.0, float(os.getenv("FLASHBULLJEV_RPS", "0")))
    except ValueError:
        return 0.0


def check_auth(headers: Any) -> bool:
    """True when the request carries the right Bearer token (or auth is off)."""
    key = _api_key()
    if not key:
        return True
    auth = str(headers.get("authorization", ""))
    return hmac.compare_digest(auth, f"Bearer {key}")


def check_rate(ip: str) -> bool:
    """Sliding-window per-IP limiter. True = allowed."""
    rps = _rate_limit()
    if rps <= 0:
        return True
    now = time.monotonic()
    window = 1.0
    with _rate_lock:
        hits = [t for t in _rate_hits.get(ip, []) if now - t < window]
        if len(hits) >= rps:
            _rate_hits[ip] = hits
            return False
        hits.append(now)
        _rate_hits[ip] = hits
        return True


def make_engine() -> FlashBullJevEngine:
    """Build the engine from env: FLASHBULLJEV_BACKEND, FLASHBULLJEV_MODEL, TEMP, CALIB."""
    backend = get_backend(os.getenv("FLASHBULLJEV_BACKEND", "fake"))
    # custom Ollama model via env
    model_env = os.getenv("FLASHBULLJEV_MODEL", "")
    if model_env and hasattr(backend, "model"):
        backend.model = model_env
    temp = float(os.getenv("FLASHBULLJEV_TEMP", "1.0"))
    calib_path = os.getenv("FLASHBULLJEV_CALIB", "")
    if calib_path:
        temp = load_temperature(calib_path)
    return FlashBullJevEngine(backend=backend, temperature=temp)


_engine = make_engine()


class DecideRequest(BaseModel):
    state: Any
    questions: Dict[str, Any] = {}
    model: str = "flashbulljev-latest"
    backend: str = ""
    temperature: float | None = None


def engine_for(req: DecideRequest) -> FlashBullJevEngine:
    """Return a dedicated engine when the request asks for another backend."""
    if req.backend:
        b = get_backend(req.backend)
        t = float(req.temperature) if req.temperature is not None else _engine.temperature
        return FlashBullJevEngine(backend=b, temperature=t)
    if req.temperature is not None:
        return FlashBullJevEngine(backend=_engine.backend, temperature=float(req.temperature))
    return _engine


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health status with cache stats and backend info."""
    return {
        "model": MODEL_ID,
        "status": "ok",
        "memory": _engine.memory.stats(),
        "backend": getattr(_engine.backend, "name", "fake"),
        "backend_model": getattr(_engine.backend, "model", ""),
        "temperature": _engine.temperature,
    }


@app.post("/v1/decisions")
def decisions(req: DecideRequest, request: Request) -> Dict[str, Any]:
    """Native flashbulljev endpoint."""
    if not check_auth(request.headers):
        raise HTTPException(status_code=401, detail="invalid API key")
    if not check_rate(request.client.host if request.client else "?"):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    eng = engine_for(req)
    res = run_pipeline(eng, req.state, req.questions)
    return {
        "model": MODEL_ID,
        "answers": res.answers,
        "gf": res.gf,
        "cache": {"hit": res.cache_hit, "latency_ms": res.latency_ms, "backend": getattr(eng.backend, "name", "fake")},
        "usage": {"input_tokens": len(str(req.state)) // 4, "output_tokens": 0},
    }


@app.post("/v1/systemone")
def systemone(req: DecideRequest, request: Request) -> Dict[str, Any]:
    """Jev-compatible System One endpoint."""
    if not check_auth(request.headers):
        raise HTTPException(status_code=401, detail="invalid API key")
    if not check_rate(request.client.host if request.client else "?"):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    eng = engine_for(req)
    raw = eng.decide(req.state, req.questions)
    answers: Dict[str, Any] = dict(raw.get("answers", {}))
    return {
        "model": MODEL_ID,
        "answers": answers,
        "usage": {"input_tokens": len(str(req.state)) // 4, "output_tokens": 0},
        "x_flash": {"gf": raw.get("gf", {}), "cache": raw.get("cache", {}), "backend": getattr(eng.backend, "name", "fake")},
    }


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    """List available models."""
    return {"models": [{"id": MODEL_ID}, {"id": "flashbulljev-latest"}, {"id": "jev-flash-alias"}]}
