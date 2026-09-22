"""Clean pipeline: ingest -> route -> decide -> certify -> emit.

Each stage is a pure function. Lists/arrays, clean code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .engine import FlashBullJevEngine


@dataclass
class PipelineResult:
    """Pipeline result with stage log."""

    answers: Dict[str, Any] = field(default_factory=dict)
    gf: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    cache_hit: bool = False
    steps: List[str] = field(default_factory=list)


def ingest(state: Any) -> Dict[str, Any]:
    """Normalize input into a dict. Pure function."""
    if isinstance(state, dict):
        return dict(state)
    return {"state": state}


def route_questions(questions: Dict[str, Any]) -> List[str]:
    """Return the list of question keys to run."""
    keys: List[str] = list(questions.keys())
    return [k for k in keys if isinstance(k, str)]


def decide_step(engine: FlashBullJevEngine, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
    """Call the fast+stored engine."""
    return engine.decide(state, questions)


def certify_step(result: Dict[str, Any]) -> bool:
    """True when the fragment is certified or the cache hits."""
    if bool(result.get("cache", {}).get("hit", False)):
        return True
    gf: Dict[str, Any] = dict(result.get("gf", {}))
    return bool(gf.get("certified", gf.get("certificato", False)))


def emit_step(result: Dict[str, Any]) -> Dict[str, Any]:
    """Build the final clean output."""
    answers: Dict[str, Any] = dict(result.get("answers", {}))
    keys: List[str] = list(answers.keys())
    clean: Dict[str, Any] = {k: answers[k] for k in keys}
    return {
        "model": result.get("model", "flashbulljev"),
        "answers": clean,
        "gf": result.get("gf", {}),
        "cache": result.get("cache", {}),
        "usage": result.get("usage", {"input_tokens": 0, "output_tokens": 0}),
    }


def run_pipeline(engine: FlashBullJevEngine, state: Any, questions: Dict[str, Any]) -> PipelineResult:
    """Run the full 5-stage pipeline."""
    steps: List[str] = []
    norm = ingest(state)
    steps.append("ingest")
    qkeys = route_questions(questions)
    steps.append(f"route:{len(qkeys)}")
    raw = decide_step(engine, norm.get("state", state), questions)
    steps.append("decide:hit" if raw.get("cache", {}).get("hit") else "decide:miss")
    ok = certify_step(raw)
    steps.append(f"certify:{ok}")
    out = emit_step(raw)
    steps.append("emit")
    return PipelineResult(
        answers=out["answers"],
        gf=out["gf"],
        latency_ms=float(out.get("cache", {}).get("latency_ms", 0.0)),
        cache_hit=bool(out.get("cache", {}).get("hit", False)),
        steps=steps,
    )


def batch_states(states: List[Any]) -> List[Dict[str, Any]]:
    """Turn a list of states into normalized dicts. Arrays/lists compliant."""
    out: List[Dict[str, Any]] = []
    for s in list(states):
        out.append(ingest(s))
    return out
