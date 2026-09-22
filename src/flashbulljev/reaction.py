"""Rapid-fire reaction battery: 16 quick input -> output probes.

Each case fires ONE question at the model (fast) and checks the reaction:
yes/no (noul), department (choice), frustration (score), fill level (numeric).
Used by `python run.py react` for live Q&A benchmarking.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from .backends import get_backend
from .engine import FlashBullJevEngine
from .pipeline import run_pipeline


def reaction_cases() -> List[Dict[str, Any]]:
    """Labeled rapid-fire cases. Pure data, lists only."""
    return [
        {"id": "u01", "state": "URGENT: server down, all payments failing!",
         "question": {"type": "noul", "instructions": "Does this convey urgency?"},
         "expect": {"kind": "bool_yes"}},
        {"id": "u02", "state": "thanks, all good, see you tomorrow",
         "question": {"type": "noul", "instructions": "Does this convey urgency?"},
         "expect": {"kind": "bool_no"}},
        {"id": "u03", "state": "Reminder: your invoice is due next week",
         "question": {"type": "noul", "instructions": "Does this convey urgency?"},
         "expect": {"kind": "bool_no"}},
        {"id": "u04", "state": "HELP! database deleted, customers are furious",
         "question": {"type": "noul", "instructions": "Does this convey urgency?"},
         "expect": {"kind": "bool_yes"}},
        {"id": "u05", "state": "hi, a quick question about pricing",
         "question": {"type": "noul", "instructions": "Does this convey urgency?"},
         "expect": {"kind": "bool_no"}},
        {"id": "u06", "state": "CRITICAL outage, refund everyone NOW",
         "question": {"type": "noul", "instructions": "Does this convey urgency?"},
         "expect": {"kind": "bool_yes"}},
        {"id": "d01", "state": "my card was charged twice, I need a refund",
         "question": {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": {"billing": "payments and refunds", "technical": "bugs and outages", "sales": "pricing and deals"}},
         "expect": {"kind": "choice", "value": "billing"}},
        {"id": "d02", "state": "the app crashes on login since the update",
         "question": {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": {"billing": "payments and refunds", "technical": "bugs and outages", "sales": "pricing and deals"}},
         "expect": {"kind": "choice", "value": "technical"}},
        {"id": "d03", "state": "do you offer discounts for large teams?",
         "question": {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": {"billing": "payments and refunds", "technical": "bugs and outages", "sales": "pricing and deals"}},
         "expect": {"kind": "choice", "value": "sales"}},
        {"id": "d04", "state": "the API returns 500 on POST /pay",
         "question": {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": {"billing": "payments and refunds", "technical": "bugs and outages", "sales": "pricing and deals"}},
         "expect": {"kind": "choice", "value": "technical"}},
        {"id": "f01", "state": "thanks for the quick fix, great job!",
         "question": {"type": "score", "instructions": "How frustrated is the customer?",
                      "criteria": ["Calm", "Frustrated", "Very angry"]},
         "expect": {"kind": "score", "value": 0}},
        {"id": "f02", "state": "this is the third time, I am losing patience",
         "question": {"type": "score", "instructions": "How frustrated is the customer?",
                      "criteria": ["Calm", "Frustrated", "Very angry"]},
         "expect": {"kind": "score", "value": 1}},
        {"id": "f03", "state": "UNACCEPTABLE! I am leaving and telling everyone",
         "question": {"type": "score", "instructions": "How frustrated is the customer?",
                      "criteria": ["Calm", "Frustrated", "Very angry"]},
         "expect": {"kind": "score", "value": 2}},
        {"id": "n01", "state": "the tank is at three quarters",
         "question": {"type": "numeric", "instructions": "Read the fill percentage.", "unit": "percent",
                      "anchors": [{"value": 0, "description": "empty"}, {"value": 50, "description": "half"}, {"value": 100, "description": "full"}]},
         "expect": {"kind": "range", "low": 60.0, "high": 90.0}},
        {"id": "n02", "state": "the glass is half full",
         "question": {"type": "numeric", "instructions": "Read the fill percentage.", "unit": "percent",
                      "anchors": [{"value": 0, "description": "empty"}, {"value": 50, "description": "half"}, {"value": 100, "description": "full"}]},
         "expect": {"kind": "range", "low": 30.0, "high": 70.0}},
        {"id": "n03", "state": "empty tank, zero fuel left",
         "question": {"type": "numeric", "instructions": "Read the fill percentage.", "unit": "percent",
                      "anchors": [{"value": 0, "description": "empty"}, {"value": 50, "description": "half"}, {"value": 100, "description": "full"}]},
         "expect": {"kind": "range", "low": 0.0, "high": 20.0}},
    ]


def check_case(answer: Dict[str, Any], expect: Dict[str, Any]) -> bool:
    """Score one reaction against its expectation. Pure function."""
    kind = str(expect.get("kind", ""))
    if kind == "bool_yes":
        return answer.get("noul") is not None and float(answer.get("noul", 0.0)) > 0.5
    if kind == "bool_no":
        return answer.get("noul") is not None and float(answer.get("noul", 1.0)) < 0.5
    if kind == "choice":
        return answer.get("choice") == expect.get("value")
    if kind == "score":
        probs = dict(answer.get("probabilities", {}))
        if not probs:
            return False
        best = max(probs.keys(), key=lambda k: probs[k])
        return int(best) == int(expect.get("value", -1))
    if kind == "range":
        v = float(answer.get("value", -1.0))
        return float(expect.get("low", 0.0)) <= v <= float(expect.get("high", 0.0))
    return False


def check_contract(answer: Dict[str, Any]) -> List[str]:
    """Validate the output contract. Returns a list of violations (empty = OK)."""
    problems: List[str] = []
    if "type" not in answer:
        problems.append("missing type")
    if "status" not in answer:
        problems.append("missing status")
    if "confidence" not in answer:
        problems.append("missing confidence")
    else:
        c = float(answer["confidence"])
        if not 0.0 <= c <= 1.0:
            problems.append(f"confidence out of range: {c}")
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and probs:
        s = sum(float(v) for v in probs.values())
        if abs(s - 1.0) > 1e-4:
            problems.append(f"probabilities sum to {s}")
    return problems


def run_reaction(backend_name: str = "", save: bool = True) -> Dict[str, Any]:
    """Fire the whole battery. Returns results + summary dict."""
    backend = get_backend(backend_name or os.getenv("FLASHBULLJEV_BACKEND", "fake"))
    eng = FlashBullJevEngine(backend=backend)
    cases = reaction_cases()
    rows: List[Dict[str, Any]] = []
    for case in cases:
        t0 = time.perf_counter()
        res = run_pipeline(eng, case["state"], {"q": case["question"]})
        dt_ms = (time.perf_counter() - t0) * 1000.0
        ans = dict(res.answers.get("q", {}))
        ok = check_case(ans, dict(case["expect"]))
        contract = check_contract(ans)
        rows.append({"id": case["id"], "ok": ok, "ms": round(dt_ms, 1),
                     "contract_ok": len(contract) == 0, "contract": contract,
                     "answer": ans, "expect": case["expect"], "state": case["state"]})
    hits = sum(1 for r in rows if r["ok"])
    lat = [r["ms"] for r in rows]
    summary: Dict[str, Any] = {
        "backend": getattr(backend, "name", "?"),
        "model": getattr(backend, "model", ""),
        "n": len(rows),
        "correct": hits,
        "accuracy": round(hits / max(len(rows), 1), 3),
        "contract_ok": sum(1 for r in rows if r["contract_ok"]),
        "avg_ms": round(sum(lat) / max(len(lat), 1), 1),
        "cache": eng.memory.stats(),
    }
    out = {"summary": summary, "rows": rows}
    if save:
        os.makedirs("output", exist_ok=True)
        path = f"output/reaction_{getattr(backend, 'name', 'fake')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return out


def print_reaction(out: Dict[str, Any]) -> None:
    """Print the rapid-fire table."""
    s = out["summary"]
    print(f"backend={s['backend']} model={s['model']} accuracy={s['correct']}/{s['n']}={s['accuracy']} avg={s['avg_ms']}ms contract={s['contract_ok']}/{s['n']}")
    for r in out["rows"]:
        mark = "OK " if r["ok"] else "KO "
        cmark = "" if r["contract_ok"] else f" CONTRACT:{r['contract']}"
        print(f"{mark} {r['id']} {r['ms']:>8}ms {str(r['answer'].get('noul', r['answer'].get('choice', r['answer'].get('score', r['answer'].get('value'))))):>10}{cmark}  <- {r['state'][:48]}")
