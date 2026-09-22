"""Rapid-fire reaction battery: 52 labeled input -> output probes.

Groups: urgency (12), sentiment (8), department (12), frustration (12),
fill level (8). Each case fires ONE question (fast) and checks the reaction.
Split: every 3rd case per group -> eval, rest -> fit (real calibration set).
Used by `python run.py react` and `python run.py react-calib`.
"""
from __future__ import annotations

import datetime
import json
import os
import time
from typing import Any, Dict, List, Tuple

from .backends import get_backend
from .calibration import apply_temperature, brier, ece, fit_temperature, nll
from .engine import FlashBullJevEngine
from .pipeline import run_pipeline
from .prompts import build_boolean_prompt, build_mc_prompt, build_score_prompt

DEPT_CRITERIA = {"billing": "payments and refunds", "technical": "bugs and outages", "sales": "pricing and deals"}
SCORE_LEVELS = ["Calm", "Frustrated", "Very angry"]
FILL_ANCHORS = [{"value": 0, "description": "empty"}, {"value": 50, "description": "half"}, {"value": 100, "description": "full"}]
FILL_Q = {"type": "numeric", "instructions": "Read the fill percentage.", "unit": "percent", "anchors": FILL_ANCHORS}
DEPT_Q = {"type": "choice", "instructions": "Which team should handle this?", "criteria": DEPT_CRITERIA}
SCORE_Q = {"type": "score", "instructions": "How frustrated is the customer?", "criteria": SCORE_LEVELS}
URGENT_Q = {"type": "noul", "instructions": "Does this convey urgency?"}
POSITIVE_Q = {"type": "noul", "instructions": "Is the sentiment positive?"}


def reaction_cases() -> List[Dict[str, Any]]:
    """52 labeled rapid-fire cases. Pure data, lists only."""
    cases: List[Dict[str, Any]] = []
    # urgency: 6 yes / 6 no
    for i, s in enumerate(["URGENT: server down, all payments failing!",
                           "HELP! database deleted, customers are furious",
                           "CRITICAL outage, refund everyone NOW",
                           "data breach, passwords leaked, act now",
                           "production is on fire, need you ASAP",
                           "my account was hacked this morning"]):
        cases.append({"id": f"u{i+1:02d}", "state": s, "question": URGENT_Q, "expect": {"kind": "bool_yes"}})
    for i, s in enumerate(["thanks, all good, see you tomorrow",
                           "Reminder: your invoice is due next week",
                           "hi, a quick question about pricing",
                           "just checking the docs, no rush",
                           "love the new feature, well done",
                           "see you at the meeting tomorrow"]):
        cases.append({"id": f"u{i+7:02d}", "state": s, "question": URGENT_Q, "expect": {"kind": "bool_no"}})
    # sentiment: 4 positive / 4 negative
    for i, s in enumerate(["absolutely love it, best tool ever",
                           "great support, solved in minutes",
                           "works perfectly, thank you",
                           "five stars, highly recommended"]):
        cases.append({"id": f"s{i+1:02d}", "state": s, "question": POSITIVE_Q, "expect": {"kind": "bool_yes"}})
    for i, s in enumerate(["terrible experience, never again",
                           "this update ruined everything",
                           "worst support I have ever seen",
                           "complete waste of money"]):
        cases.append({"id": f"s{i+5:02d}", "state": s, "question": POSITIVE_Q, "expect": {"kind": "bool_no"}})
    # department: 4 billing / 4 technical / 4 sales
    billing = ["my card was charged twice, I need a refund",
               "the invoice has the wrong VAT amount",
               "please send the receipt for my last payment",
               "cancel my subscription and refund this month"]
    technical = ["the app crashes on login since the update",
                 "the API returns 500 on POST /pay",
                 "sync is stuck at 99% for hours",
                 "my microphone is not detected on calls"]
    sales = ["do you offer discounts for large teams?",
             "do you have an enterprise plan?",
             "can I trial premium for a month?",
             "I need a quote for 200 seats"]
    for i, (s, v) in enumerate([(x, "billing") for x in billing] + [(x, "technical") for x in technical] + [(x, "sales") for x in sales]):
        cases.append({"id": f"d{i+1:02d}", "state": s, "question": DEPT_Q, "expect": {"kind": "choice", "value": v}})
    # frustration: 4 calm / 4 frustrated / 4 angry
    calm = ["thanks for the quick fix, great job!",
            "no worries, take your time",
            "all good, appreciate the help",
            "perfect, that solved it"]
    frust = ["this is the third time, I am losing patience",
             "been waiting two days for an answer",
             "this keeps happening every week",
             "I explained this already twice"]
    angry = ["UNACCEPTABLE! I am leaving and telling everyone",
             "I want a manager NOW, this is a disgrace",
             "filing a chargeback and reporting you",
             "you lost my data, this is unforgivable"]
    for i, (s, v) in enumerate([(x, 0) for x in calm] + [(x, 1) for x in frust] + [(x, 2) for x in angry]):
        cases.append({"id": f"f{i+1:02d}", "state": s, "question": SCORE_Q, "expect": {"kind": "score", "value": v}})
    # fill level: 8 ranges
    fills = [("the tank is at three quarters", 60.0, 90.0),
             ("the glass is half full", 30.0, 70.0),
             ("empty tank, zero fuel left", 0.0, 20.0),
             ("the tank is completely full", 85.0, 100.0),
             ("about a quarter left in the bottle", 10.0, 35.0),
             ("nearly full, around ninety percent", 80.0, 100.0),
             ("just a drop left, almost empty", 0.0, 15.0),
             ("a bit more than half, sixty percent?", 45.0, 75.0)]
    for i, (s, lo, hi) in enumerate(fills):
        cases.append({"id": f"n{i+1:02d}", "state": s, "question": FILL_Q,
                      "expect": {"kind": "range", "low": lo, "high": hi}})
    return cases


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
    """Validate the output contract. Returns violations (empty = OK)."""
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


def case_label_index(case: Dict[str, Any]) -> int | None:
    """Label as class index for calibration (None for numeric ranges)."""
    expect = dict(case.get("expect", {}))
    kind = str(expect.get("kind", ""))
    if kind == "bool_yes":
        return 0  # A=Yes
    if kind == "bool_no":
        return 1  # B=No
    if kind == "choice":
        opts = list((case.get("question", {}).get("criteria", {}) or {}).keys())
        try:
            return opts.index(str(expect.get("value")))
        except ValueError:
            return None
    if kind == "score":
        return int(expect.get("value", 0))
    return None


def case_raw_logits(backend: Any, case: Dict[str, Any]) -> Tuple[List[float] | None, int | None]:
    """Raw letter logits + label for one case (None when not classifiable)."""
    label = case_label_index(case)
    if label is None:
        return None, None
    q = dict(case.get("question", {}))
    state = case.get("state")
    t = str(q.get("type", "noul"))
    if t in ("noul", "boolean"):
        prompt = build_boolean_prompt(state, str(q.get("instructions", "")))
        n = 2
    elif t == "choice":
        opts = list((q.get("criteria", {}) or {}).keys())
        prompt = build_mc_prompt(state, str(q.get("instructions", "")), opts)
        n = len(opts)
    elif t == "score":
        levels = list(q.get("criteria", []) or [])
        prompt = build_score_prompt(state, str(q.get("instructions", "")), levels)
        n = len(levels)
    else:
        return None, None
    logits, _ = backend.logits_for(prompt, n)
    return list(logits), label


def split_cases(cases: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Per-group split: every 3rd case -> eval, rest -> fit."""
    fit: List[Dict[str, Any]] = []
    ev: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {}
    for case in cases:
        g = str(case.get("id", "x"))[:1]
        i = counters.get(g, 0)
        counters[g] = i + 1
        (ev if i % 3 == 2 else fit).append(case)
    return fit, ev


def eval_metrics(logits_list: List[List[float]], labels: List[int], temperature: float) -> Dict[str, Any]:
    """NLL/Brier/ECE/accuracy at a temperature. Pure computation."""
    import numpy as np

    probs = [apply_temperature(list(lg), temperature) for lg in logits_list]
    preds = [int(np.argmax(np.array(p))) for p in probs]
    acc = sum(1 for pr, y in zip(preds, labels) if pr == y) / max(len(labels), 1)
    return {"temperature": temperature, "nll": round(nll(probs, labels), 4),
            "brier": round(brier(probs, labels), 4), "ece": round(ece(probs, labels), 4),
            "accuracy": round(acc, 3), "n": len(labels)}


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
        os.makedirs("results", exist_ok=True)
        path = f"results/reaction_{getattr(backend, 'name', 'fake')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return out


def run_reaction_calibration(backend_name: str = "") -> Dict[str, Any]:
    """Fit temperature on the fit split, evaluate on held-out eval split."""
    backend = get_backend(backend_name or os.getenv("FLASHBULLJEV_BACKEND", "fake"))
    fit_cases, eval_cases = split_cases(reaction_cases())
    fit_lg: List[List[float]] = []
    fit_y: List[int] = []
    for case in fit_cases:
        lg, y = case_raw_logits(backend, case)
        if lg is not None and y is not None:
            fit_lg.append(lg)
            fit_y.append(y)
    eval_lg: List[List[float]] = []
    eval_y: List[int] = []
    for case in eval_cases:
        lg, y = case_raw_logits(backend, case)
        if lg is not None and y is not None:
            eval_lg.append(lg)
            eval_y.append(y)
    temp, fit_report = fit_temperature(fit_lg, fit_y)
    out: Dict[str, Any] = {
        "backend": getattr(backend, "name", "?"),
        "model": getattr(backend, "model", ""),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fit_ids": [c["id"] for c in fit_cases],
        "eval_ids": [c["id"] for c in eval_cases],
        "fit": {"n": len(fit_y), "temperature": temp, "report": fit_report},
        "eval_T1": eval_metrics(eval_lg, eval_y, 1.0),
        "eval_Tfit": eval_metrics(eval_lg, eval_y, temp),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/reaction_calibration.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open("calibration-fit.json", "w", encoding="utf-8") as f:
        json.dump({"temperature": temp, "report": fit_report,
                   "meta": {"backend": out["backend"], "model": out["model"], "n_fit": len(fit_y)}}, f, indent=2)
    return out


def print_reaction(out: Dict[str, Any]) -> None:
    """Print the rapid-fire table."""
    s = out["summary"]
    print(f"backend={s['backend']} model={s['model']} accuracy={s['correct']}/{s['n']}={s['accuracy']} avg={s['avg_ms']}ms contract={s['contract_ok']}/{s['n']}")
    for r in out["rows"]:
        mark = "OK " if r["ok"] else "KO "
        cmark = "" if r["contract_ok"] else f" CONTRACT:{r['contract']}"
        print(f"{mark} {r['id']} {r['ms']:>8}ms {str(r['answer'].get('noul', r['answer'].get('choice', r['answer'].get('score', r['answer'].get('value'))))):>10}{cmark}  <- {r['state'][:48]}")


def print_calibration(out: Dict[str, Any]) -> None:
    """Print the fit/eval calibration report."""
    print(f"backend={out['backend']} model={out['model']}")
    print(f"fit n={out['fit']['n']} T={out['fit']['temperature']} {out['fit']['report']}")
    print(f"eval T=1.0 : {out['eval_T1']}")
    print(f"eval T=fit : {out['eval_Tfit']}")
