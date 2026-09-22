"""FastAPI service with a Jev-compatible interface and real/fake backends."""
from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from .backends import get_backend
from .calibration import load_temperature
from .engine import MODEL_ID, FlashBullJevEngine
from .pipeline import run_pipeline

app = FastAPI(title="flashbulljev", version="0.3.0")


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
def decisions(req: DecideRequest) -> Dict[str, Any]:
    """Native flashbulljev endpoint."""
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
def systemone(req: DecideRequest) -> Dict[str, Any]:
    """Jev-compatible System One endpoint."""
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
