"""flashbulljev engine: real prototype with fake + Ollama logit-only + GF cache.

Real miss ~650ms with Ollama Qwen 2B, hit ~0.05ms with FragmentMemory.
Pure functions, arrays, lists, parallel execution.
"""
from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np

from .backends import FakeBackend, get_backend, softmax_list
from .calibration import apply_temperature
from .memory_gf import FragmentMemory, certify_fragment, state_to_vector
from .prompts import build_boolean_prompt, build_mc_prompt, build_numeric_prompt, build_score_prompt
from .schemas import (
    BooleanQuestion,
    ChoiceQuestion,
    NumericQuestion,
    ScoreQuestion,
    parse_questions,
)

MODEL_ID = "flashbulljev-0.4.0"


def softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
    """Stable softmax over a list."""
    return softmax_list(list(logits), temperature)


def confidence_from_probs(probs: List[float]) -> float:
    """Confidence score (n*pmax-1)/(n-1)."""
    n = len(probs)
    if n <= 1:
        return 1.0
    pmax = max(probs)
    return float((n * pmax - 1.0) / (n - 1.0))


def fake_logits(state: Any, qkey: str, n_options: int) -> List[float]:
    """Backwards-compatible deterministic fake logits."""
    h = hashlib.sha256(f"{state!r}||{qkey}".encode()).digest()
    logits: List[float] = []
    for i in range(max(n_options, 2)):
        logits.append(float(h[(i * 3) % len(h)] + h[(i * 7 + 1) % len(h)] / 255.0))
    return logits


def decide_boolean(q: BooleanQuestion, state: Any) -> Dict[str, Any]:
    """Backwards-compatible fake boolean decision."""
    logits = fake_logits(state, q.instructions, 2)
    probs = softmax(logits)
    p_true = float(probs[0])
    if q.allow_abstain and max(probs) < 0.55:
        return {"type": "noul", "noul": None, "status": "insufficient_evidence", "confidence": 0.0}
    return {"type": "noul", "noul": p_true, "status": "ok", "confidence": confidence_from_probs(probs)}


def decide_choice(q: ChoiceQuestion, state: Any) -> Dict[str, Any]:
    """Backwards-compatible fake choice decision."""
    opts: List[str] = q.option_ids() or ["a", "b"]
    logits = fake_logits(state, q.instructions + "".join(opts), len(opts))
    probs = softmax(logits)
    idx = int(np.argmax(np.array(probs)))
    prob_dict = {opts[i]: float(probs[i]) for i in range(len(opts))}
    if q.allow_abstain and max(probs) < 0.45:
        return {"type": "choice", "choice": None, "probabilities": prob_dict, "status": "uncertain", "confidence": 0.0}
    return {"type": "choice", "choice": opts[idx], "probabilities": prob_dict, "status": "ok", "confidence": confidence_from_probs(probs)}


def decide_score(q: ScoreQuestion, state: Any) -> Dict[str, Any]:
    """Backwards-compatible fake score decision."""
    levels: List[str] = list(q.levels) or ["low", "high"]
    logits = fake_logits(state, q.instructions, len(levels))
    probs = softmax(logits)
    score = float(sum(i * p for i, p in enumerate(probs)))
    norm = float(score / max(len(levels) - 1, 1))
    return {"type": "score", "score": score, "normalized_score": norm,
            "probabilities": {str(i): float(probs[i]) for i in range(len(levels))},
            "legend": {str(i): levels[i] for i in range(len(levels))},
            "status": "ok", "confidence": confidence_from_probs(probs)}


def decide_numeric(q: NumericQuestion, state: Any) -> Dict[str, Any]:
    """Backwards-compatible fake numeric decision."""
    anchors: List[float] = list(q.anchors) or [0.0, 1.0]
    logits = fake_logits(state, q.instructions, len(anchors))
    probs = softmax(logits)
    value = float(sum(a * p for a, p in zip(anchors, probs)))
    return {"type": "numeric", "value": value,
            "probabilities": {str(anchors[i]): float(probs[i]) for i in range(len(anchors))},
            "unit": q.unit, "status": "ok", "confidence": confidence_from_probs(probs)}


def decide_one(qkey: str, qobj: Any, state: Any) -> Tuple[str, Dict[str, Any]]:
    """Backwards-compatible fake single decision."""
    if isinstance(qobj, BooleanQuestion):
        return qkey, decide_boolean(qobj, state)
    if isinstance(qobj, ChoiceQuestion):
        return qkey, decide_choice(qobj, state)
    if isinstance(qobj, ScoreQuestion):
        return qkey, decide_score(qobj, state)
    if isinstance(qobj, NumericQuestion):
        return qkey, decide_numeric(qobj, state)
    return qkey, {"type": "noul", "noul": 0.5, "status": "ok", "confidence": 0.0}


def decide_many(state: Any, raw_questions: Dict[str, Any], max_workers: int = 4) -> Dict[str, Any]:
    """Backwards-compatible fake parallel decisions."""
    parsed = parse_questions(raw_questions)
    keys: List[str] = list(parsed.keys())
    answers: Dict[str, Any] = {}
    _prefill = state_to_vector(state)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(decide_one, k, parsed[k], state): k for k in keys}
        for f in futs:
            k, ans = f.result()
            answers[k] = ans
    return answers


def _abstain_thresholds() -> Tuple[float, float]:
    """(tau_abs, tau_margin) from env, defaults from live tuning grid."""
    try:
        tau_abs = float(os.getenv("FLASHBULLJEV_ABSTAIN_TAU", "0.6"))
    except ValueError:
        tau_abs = 0.6
    try:
        tau_margin = float(os.getenv("FLASHBULLJEV_ABSTAIN_MARGIN", "0.3"))
    except ValueError:
        tau_margin = 0.3
    return tau_abs, tau_margin


def _should_abstain(probs: List[float], allow: bool) -> bool:
    """Unified rule: abstain when top prob or margin is below tuned thresholds."""
    if not allow or not probs:
        return False
    tau_abs, tau_margin = _abstain_thresholds()
    ordered = sorted(probs, reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    return bool(ordered[0] < tau_abs or margin < tau_margin)


def decide_one_real(qkey: str, qobj: Any, state: Any, backend: Any, temperature: float = 1.0) -> Tuple[str, Dict[str, Any], float]:
    """Single decision with a real logit-only backend. Returns (key, answer, latency)."""
    if isinstance(qobj, BooleanQuestion):
        prompt = build_boolean_prompt(state, qobj.instructions)
        labels: List[str] = ["Yes", "No"]
    elif isinstance(qobj, ChoiceQuestion):
        opts: List[str] = qobj.option_ids() or ["a", "b"]
        prompt = build_mc_prompt(state, qobj.instructions, opts)
        labels = opts
    elif isinstance(qobj, ScoreQuestion):
        levels: List[str] = list(qobj.levels) or ["low", "high"]
        prompt = build_score_prompt(state, qobj.instructions, levels)
        labels = levels
    elif isinstance(qobj, NumericQuestion):
        anchors: List[float] = list(qobj.anchors) or [0.0, 1.0]
        prompt = build_numeric_prompt(state, qobj.instructions, anchors, list(qobj.descriptions))
        labels = [str(a) for a in anchors]
    else:
        prompt = build_boolean_prompt(state, "decide")
        labels = ["Yes", "No"]
    logits, lat = backend.logits_for(prompt, len(labels))
    probs = apply_temperature(list(logits), temperature)
    conf = confidence_from_probs(probs)
    idx = int(np.argmax(np.array(probs)))
    if isinstance(qobj, BooleanQuestion):
        p_true = float(probs[0])
        if _should_abstain(probs, qobj.allow_abstain):
            return qkey, {"type": "noul", "noul": None, "status": "insufficient_evidence", "confidence": 0.0, "backend": getattr(backend, "name", "unk")}, lat
        return qkey, {"type": "noul", "noul": p_true, "status": "ok", "confidence": conf, "backend": getattr(backend, "name", "unk")}, lat
    if isinstance(qobj, ChoiceQuestion):
        opts2: List[str] = qobj.option_ids() or ["a", "b"]
        prob_dict = {opts2[i]: float(probs[i]) for i in range(len(opts2))}
        if _should_abstain(probs, qobj.allow_abstain):
            return qkey, {"type": "choice", "choice": None, "probabilities": prob_dict, "status": "uncertain", "confidence": 0.0, "backend": getattr(backend, "name", "unk")}, lat
        return qkey, {"type": "choice", "choice": opts2[idx], "probabilities": prob_dict, "status": "ok", "confidence": conf, "backend": getattr(backend, "name", "unk")}, lat
    if isinstance(qobj, ScoreQuestion):
        levels2: List[str] = list(qobj.levels) or ["low", "high"]
        if _should_abstain(probs, qobj.allow_abstain):
            return qkey, {"type": "score", "score": None, "normalized_score": None,
                          "probabilities": {str(i): float(probs[i]) for i in range(len(levels2))},
                          "legend": {str(i): levels2[i] for i in range(len(levels2))},
                          "status": "uncertain", "confidence": 0.0, "backend": getattr(backend, "name", "unk")}, lat
        score = float(sum(i * p for i, p in enumerate(probs)))
        norm = float(score / max(len(levels2) - 1, 1))
        return qkey, {"type": "score", "score": score, "normalized_score": norm,
                      "probabilities": {str(i): float(probs[i]) for i in range(len(levels2))},
                      "legend": {str(i): levels2[i] for i in range(len(levels2))},
                      "status": "ok", "confidence": conf, "backend": getattr(backend, "name", "unk")}, lat
    if isinstance(qobj, NumericQuestion):
        anchors2: List[float] = list(qobj.anchors) or [0.0, 1.0]
        value = float(sum(a * p for a, p in zip(anchors2, probs)))
        return qkey, {"type": "numeric", "value": value,
                      "probabilities": {str(anchors2[i]): float(probs[i]) for i in range(len(anchors2))},
                      "unit": qobj.unit, "status": "ok", "confidence": conf, "backend": getattr(backend, "name", "unk")}, lat
    return qkey, {"type": "noul", "noul": 0.5, "status": "ok", "confidence": 0.0}, lat


def decide_many_real(state: Any, raw_questions: Dict[str, Any], backend: Any, temperature: float = 1.0, max_workers: int = 4) -> Tuple[Dict[str, Any], float]:
    """Real parallel decisions. Returns (answers, total_backend_ms)."""
    parsed = parse_questions(raw_questions)
    keys: List[str] = list(parsed.keys())
    answers: Dict[str, Any] = {}
    lat_list: List[float] = []
    _prefill = state_to_vector(state)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(decide_one_real, k, parsed[k], state, backend, temperature): k for k in keys}
        for f in futs:
            k, ans, lat = f.result()
            answers[k] = ans
            lat_list.append(float(lat))
    return answers, float(sum(lat_list))


class FlashBullJevEngine:
    """Decision engine with a pluggable backend and the GF cache."""

    def __init__(self, backend: Any | None = None, backend_name: str = "", temperature: float = 1.0) -> None:
        self.memory = FragmentMemory()
        self.model = MODEL_ID
        self.backend = backend or get_backend(backend_name)
        if isinstance(self.backend, type):
            self.backend = self.backend()
        self.temperature = float(temperature)
        self._is_fake = getattr(self.backend, "name", "fake") == "fake"
        self.cache_db = os.getenv("FLASHBULLJEV_CACHE_DB", "")
        if self.cache_db:
            self.memory.attach_db(self.cache_db)

    def decide(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Decide with cache: ~1ms on hit, real backend latency on miss."""
        t0 = time.perf_counter()
        qkeys: List[str] = list(questions.keys())
        # the key includes the backend so fake and real entries never mix
        bname = str(getattr(self.backend, "name", "fake") + ":" + getattr(self.backend, "model", ""))
        key = self.memory.key(f"{state}||{bname}", qkeys)
        hit, cached = self.memory.recall(key)
        if hit and cached is not None:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            out = dict(cached)
            out["cache"] = {"hit": True, "latency_ms": dt_ms}
            out["usage"] = {"input_tokens": len(str(state)) // 4, "output_tokens": 0}
            return out
        vec = state_to_vector(state)
        fo = vec.copy()
        fx = vec * 0.98 + 0.02 * float(np.mean(vec))
        fy = (vec - float(np.mean(vec))) * 0.05
        cert = certify_fragment(fo, fx, fy, fo)
        if self._is_fake:
            answers = decide_many(state, questions)
            time.sleep(0.05)
            backend_ms = 50.0
        else:
            answers, backend_ms = decide_many_real(state, questions, self.backend, self.temperature)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        result: Dict[str, Any] = {
            "model": f"{self.model}+{getattr(self.backend, 'name', 'fake')}:{getattr(self.backend, 'model', '')}",
            "answers": answers,
            "gf": cert,
            "cache": {"hit": False, "latency_ms": dt_ms, "backend_ms": backend_ms},
            "usage": {"input_tokens": len(str(state)) // 4, "output_tokens": 0},
        }
        if bool(cert.get("certified", cert.get("certificato", False))):
            self.memory.save(key, result)
        return result
