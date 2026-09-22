"""Rapid-fire reaction battery: 576 labeled input -> output probes.

Base: 96 cases in 5 groups (urgency 20, sentiment 16, department 24,
frustration 20, fill 16) x 6 label-preserving variants = 576.
Split: whole families (base + variants) with family_idx % 3 == 2 -> eval,
rest -> fit. The battery doubles as a real calibration set.
Extra probes: abstention (12 vague cases), position bias (option reversal),
JSON baseline (free-text generation vs logit-only).
"""
from __future__ import annotations

import datetime
import json
import os
import re
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


def _base_cases() -> List[Dict[str, Any]]:
    """52 curated base cases."""
    cases: List[Dict[str, Any]] = []
    for i, s in enumerate(["URGENT: server down, all payments failing!",
                           "HELP! database deleted, customers are furious",
                           "CRITICAL outage, refund everyone NOW",
                           "data breach, passwords leaked, act now",
                           "production is on fire, need you ASAP",
                           "my account was hacked this morning",
                           "SECURITY ALERT: unauthorized login from unknown device",
                           "payment gateway down, losing orders every minute",
                           "fire alarm in building B, evacuate now",
                           "my flight leaves in one hour and the ticket is wrong"]):
        cases.append({"id": f"u{i+1:02d}", "state": s, "question": URGENT_Q, "expect": {"kind": "bool_yes"}})
    for i, s in enumerate(["thanks, all good, see you tomorrow",
                           "Reminder: your invoice is due next week",
                           "hi, a quick question about pricing",
                           "just checking the docs, no rush",
                           "love the new feature, well done",
                           "see you at the meeting tomorrow",
                           "just wanted to say the new design looks great",
                           "can we move Friday's call to Monday?",
                           "where can I download last month's report?",
                           "happy birthday! hope you have a great day"]):
        cases.append({"id": f"u{i+11:02d}", "state": s, "question": URGENT_Q, "expect": {"kind": "bool_no"}})
    for i, s in enumerate(["absolutely love it, best tool ever",
                           "great support, solved in minutes",
                           "works perfectly, thank you",
                           "five stars, highly recommended",
                           "super fast delivery, very impressed",
                           "the new dashboard is a joy to use",
                           "support went above and beyond, thanks",
                           "finally an app that just works"]):
        cases.append({"id": f"s{i+1:02d}", "state": s, "question": POSITIVE_Q, "expect": {"kind": "bool_yes"}})
    for i, s in enumerate(["terrible experience, never again",
                           "this update ruined everything",
                           "worst support I have ever seen",
                           "complete waste of money",
                           "charged me three times and nobody replies",
                           "the export feature is broken again",
                           "rude agent, hung up on me",
                           "I regret switching to this service"]):
        cases.append({"id": f"s{i+9:02d}", "state": s, "question": POSITIVE_Q, "expect": {"kind": "bool_no"}})
    billing = ["my card was charged twice, I need a refund",
               "the invoice has the wrong VAT amount",
               "please send the receipt for my last payment",
               "cancel my subscription and refund this month",
               "why is there a $50 fee on my bill?",
               "I need a VAT invoice for accounting",
               "my coupon code is not applied at checkout",
               "you renewed without asking, I want my money back"]
    technical = ["the app crashes on login since the update",
                 "the API returns 500 on POST /pay",
                 "sync is stuck at 99% for hours",
                 "my microphone is not detected on calls",
                 "two-factor SMS never arrives",
                 "the desktop app freezes on startup",
                 "webhooks stopped firing yesterday",
                 "cannot upload files larger than 10MB"]
    sales = ["do you offer discounts for large teams?",
             "do you have an enterprise plan?",
             "can I trial premium for a month?",
             "I need a quote for 200 seats",
             "is there a startup discount program?",
             "can we schedule a sales demo?",
             "what does the pro tier include?",
             "do you price-match competitors?"]
    for i, (s, v) in enumerate([(x, "billing") for x in billing] + [(x, "technical") for x in technical] + [(x, "sales") for x in sales]):
        cases.append({"id": f"d{i+1:02d}", "state": s, "question": DEPT_Q, "expect": {"kind": "choice", "value": v}})
    calm = ["thanks for the quick fix, great job!",
            "no worries, take your time",
            "all good, appreciate the help",
            "perfect, that solved it",
            "no problem at all, thanks for checking",
            "take all the time you need",
            "you have been very helpful, grazie",
            "all sorted on my end, cheers"]
    frust = ["this is the third time, I am losing patience",
             "been waiting two days for an answer",
             "this keeps happening every week",
             "I explained this already twice"]
    angry = ["UNACCEPTABLE! I am leaving and telling everyone",
             "I want a manager NOW, this is a disgrace",
             "filing a chargeback and reporting you",
             "you lost my data, this is unforgivable",
             "I am contacting my lawyer about this",
             "this is theft, I am reporting fraud",
             "never contacting support again, useless",
             "I will cancel everything right now"]
    for i, (s, v) in enumerate([(x, 0) for x in calm] + [(x, 1) for x in frust] + [(x, 2) for x in angry]):
        cases.append({"id": f"f{i+1:02d}", "state": s, "question": SCORE_Q, "expect": {"kind": "score", "value": v}})
    fills = [("the tank is at three quarters", 60.0, 90.0),
             ("the glass is half full", 30.0, 70.0),
             ("empty tank, zero fuel left", 0.0, 20.0),
             ("the tank is completely full", 85.0, 100.0),
             ("about a quarter left in the bottle", 10.0, 35.0),
             ("nearly full, around ninety percent", 80.0, 100.0),
             ("just a drop left, almost empty", 0.0, 15.0),
             ("a bit more than half, sixty percent?", 45.0, 75.0),
             ("the bottle is full to the brim", 90.0, 100.0),
             ("a quarter tank, roughly 25", 15.0, 35.0),
             ("about eighty percent charged", 70.0, 90.0),
             ("ten percent battery, dying", 0.0, 20.0),
             ("sixty percent downloaded", 50.0, 70.0),
             ("tank is bone dry", 0.0, 10.0),
             ("halfway there, fifty-fifty", 40.0, 60.0),
             ("ninety-five percent complete", 85.0, 100.0)]
    for i, (s, lo, hi) in enumerate(fills):
        cases.append({"id": f"n{i+1:02d}", "state": s, "question": FILL_Q,
                      "expect": {"kind": "range", "low": lo, "high": hi}})
    return cases


def _variants(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Label-preserving variants of one case (original + 5 transforms)."""
    s = str(case["state"])
    out = [dict(case)]
    for suffix, text in [("b", s.upper()),
                         ("c", f"Please note: {s} Thanks."),
                         ("d", f'"{s}"'),
                         ("e", f"Quick: {s}"),
                         ("f", f"{s} please.")]:
        c = dict(case)
        c = {"id": case["id"] + suffix, "state": text, "question": case["question"],
             "expect": case["expect"], "variant_of": case["id"]}
        out.append(c)
    return out


def reaction_cases() -> List[Dict[str, Any]]:
    """576 cases: 96 base x 6 variants."""
    all_cases: List[Dict[str, Any]] = []
    for case in _base_cases():
        all_cases.extend(_variants(case))
    return all_cases


def abstain_cases() -> List[Dict[str, Any]]:
    """12 vague cases where abstaining is the right reaction."""
    states = ["the thing is stuff", "hello", "???", "update", "it depends",
              "maybe later", "hmm", "see attached (no attachment)",
              "as per my last email", "ping", "test test", "n/a"]
    questions = [
        {"type": "noul", "instructions": "Does this convey urgency?"},
        {"type": "choice", "instructions": "Which team should handle this?",
         "criteria": DEPT_CRITERIA},
    ]
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(states):
        out.append({"id": f"a{i+1:02d}", "state": s, "question": questions[i % 2]})
    return out


def check_case(answer: Dict[str, Any], expect: Dict[str, Any]) -> bool:
    """Score one reaction (abstentions count as wrong). Pure function."""
    if answer.get("status", "ok") != "ok":
        return False
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


def is_abstained(answer: Dict[str, Any]) -> bool:
    """True when the model declined to decide. Pure function."""
    if answer.get("status") not in ("ok",):
        return True
    if "noul" in answer and answer.get("noul") is None:
        return True
    if "choice" in answer and answer.get("choice") is None:
        return True
    return False


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


def _family(case: Dict[str, Any]) -> str:
    """Family id: base id without variant suffix (u01b -> u01)."""
    cid = str(case.get("id", "x"))
    return cid[:3]


def split_cases(cases: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Family split: whole families with family_idx % 3 == 2 -> eval, rest -> fit."""
    fit: List[Dict[str, Any]] = []
    ev: List[Dict[str, Any]] = []
    seen: List[str] = []
    for case in cases:
        fam = _family(case)
        if fam not in seen:
            seen.append(fam)
    groups: Dict[str, List[str]] = {}
    for fam in seen:
        groups.setdefault(fam[:1], []).append(fam)
    eval_fams = set()
    for g, fams in groups.items():
        for i, fam in enumerate(fams):
            if i % 3 == 2:
                eval_fams.add(fam)
    for case in cases:
        (ev if _family(case) in eval_fams else fit).append(case)
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


def parse_json_letter(text: str, letters: List[str]) -> str | None:
    """Extract the answer letter from free text. Prefers quoted (JSON) and standalone letters."""
    allowed = [L.upper() for L in letters]
    m = re.search(r'"([A-Za-z])"', text)
    if m and m.group(1).upper() in allowed:
        return m.group(1).upper()
    for m in re.finditer(r"\b([A-Za-z])\b", text):
        if m.group(1).upper() in allowed:
            return m.group(1).upper()
    for m in re.finditer(r"[A-Za-z]", text):
        if m.group(0).upper() in allowed:
            return m.group(0).upper()
    return None


def run_reaction(backend_name: str = "", save: bool = True, limit: int = 0) -> Dict[str, Any]:
    """Fire the battery (first `limit` cases when > 0, e.g. CI smoke)."""
    backend = get_backend(backend_name or os.getenv("FLASHBULLJEV_BACKEND", "fake"))
    eng = FlashBullJevEngine(backend=backend)
    cases = reaction_cases()
    if limit and limit > 0:
        cases = cases[:limit]
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


def case_confidence(backend: Any, state: Any, question: Dict[str, Any]) -> Tuple[float, float]:
    """(max_prob, margin) for one state+question at T=1. Pure-ish probe."""
    t = str(question.get("type", "noul"))
    if t in ("noul", "boolean"):
        prompt = build_boolean_prompt(state, str(question.get("instructions", "")))
        n = 2
    elif t == "choice":
        opts = list((question.get("criteria", {}) or {}).keys())
        prompt = build_mc_prompt(state, str(question.get("instructions", "")), opts)
        n = max(len(opts), 2)
    elif t == "score":
        levels = list(question.get("criteria", []) or [])
        prompt = build_score_prompt(state, str(question.get("instructions", "")), levels)
        n = max(len(levels), 2)
    else:
        return 0.0, 0.0
    logits, _ = backend.logits_for(prompt, n)
    probs = sorted(apply_temperature(list(logits), 1.0), reverse=True)
    top = probs[0] if probs else 0.0
    second = probs[1] if len(probs) > 1 else 0.0
    return float(top), float(top - second)


def pick_thresholds(clear: List[Tuple[float, float]], vague: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Grid-search (tau_abs, tau_margin): max vague-abstain with clear-keep >= 0.95. Pure."""
    best: Dict[str, Any] = {"tau_abs": 1.0, "tau_margin": 1.0, "vague_abstain": 0.0, "clear_keep": 0.0}
    for tau_abs in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
        for tau_margin in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
            vk = sum(1 for mx, mg in vague if mx < tau_abs or mg < tau_margin) / max(len(vague), 1)
            ck = sum(1 for mx, mg in clear if not (mx < tau_abs or mg < tau_margin)) / max(len(clear), 1)
            if ck >= 0.95 and (vk > best["vague_abstain"] or (vk == best["vague_abstain"] and ck > best["clear_keep"])):
                best = {"tau_abs": tau_abs, "tau_margin": tau_margin,
                        "vague_abstain": round(vk, 3), "clear_keep": round(ck, 3)}
    return best


def pairs_at_temperature(logits_list: List[List[float]], temperature: float) -> List[Tuple[float, float]]:
    """(max, margin) pairs from raw logits at temperature T. Pure."""
    pairs: List[Tuple[float, float]] = []
    for lg in logits_list:
        probs = sorted(apply_temperature(list(lg), temperature), reverse=True)
        top = probs[0] if probs else 0.0
        second = probs[1] if len(probs) > 1 else 0.0
        pairs.append((float(top), float(top - second)))
    return pairs


def run_abstention_tuning(backend_name: str = "") -> Dict[str, Any]:
    """Collect raw logits on clear battery + vague cases, tune thresholds at T=1 and T=3."""
    backend = get_backend(backend_name or os.getenv("FLASHBULLJEV_BACKEND", "fake"))
    clear_lg: List[List[float]] = []
    for case in reaction_cases():
        try:
            lg, y = case_raw_logits(backend, case)
        except (OSError, ValueError, RuntimeError):
            continue
        if lg is not None:
            clear_lg.append(lg)
    vague_lg: List[List[float]] = []
    for case in abstain_cases():
        try:
            lg, _ = case_raw_logits(backend, {**case, "expect": {"kind": "bool_yes"}})
        except (OSError, ValueError, RuntimeError):
            continue
        if lg is None:
            try:
                lg, _ = case_raw_logits(backend, {**case, "expect": {"kind": "choice", "value": "billing"}})
            except (OSError, ValueError, RuntimeError):
                continue
        if lg is not None:
            vague_lg.append(lg)
    tunings = {}
    for temp in (1.0, 3.0):
        tunings[f"T{temp:g}"] = pick_thresholds(pairs_at_temperature(clear_lg, temp),
                                                 pairs_at_temperature(vague_lg, temp))
    out = {"backend": getattr(backend, "name", "?"), "model": getattr(backend, "model", ""),
           "n_clear": len(clear_lg), "n_vague": len(vague_lg), "tunings": tunings}
    os.makedirs("results", exist_ok=True)
    with open(f"results/reaction_abstain_tuning_{getattr(backend, 'name', 'fake')}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def run_abstention(backend_name: str = "") -> Dict[str, Any]:
    """Fire vague cases; abstaining is correct. Returns rates + rows."""
    backend = get_backend(backend_name or os.getenv("FLASHBULLJEV_BACKEND", "fake"))
    eng = FlashBullJevEngine(backend=backend)
    rows: List[Dict[str, Any]] = []
    for case in abstain_cases():
        res = run_pipeline(eng, case["state"], {"q": case["question"]})
        ans = dict(res.answers.get("q", {}))
        rows.append({"id": case["id"], "abstained": is_abstained(ans),
                     "confidence": ans.get("confidence"), "status": ans.get("status"),
                     "state": case["state"]})
    n_abs = sum(1 for r in rows if r["abstained"])
    out = {"backend": getattr(backend, "name", "?"), "model": getattr(backend, "model", ""),
           "n": len(rows), "abstained": n_abs,
           "abstention_rate": round(n_abs / max(len(rows), 1), 3), "rows": rows}
    os.makedirs("results", exist_ok=True)
    with open(f"results/reaction_abstain_{getattr(backend, 'name', 'fake')}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def run_position_bias(backend_name: str = "") -> Dict[str, Any]:
    """Original vs reversed option order on base dept cases. Fresh engine per call (no cache)."""
    from .prompts import letters_for as _lf  # noqa: F401 (keeps import graph explicit)

    backend = get_backend(backend_name or os.getenv("FLASHBULLJEV_BACKEND", "fake"))
    bases = [c for c in _base_cases() if c["id"].startswith("d")]
    flips = 0
    ok_orig = 0
    ok_rev = 0
    rows: List[Dict[str, Any]] = []
    for case in bases:
        q = dict(case["question"])
        crit = dict(q.get("criteria", {}))
        rev_q = dict(q)
        rev_q["criteria"] = dict(reversed(list(crit.items())))
        eng1 = FlashBullJevEngine(backend=backend)
        eng2 = FlashBullJevEngine(backend=backend)
        a1 = run_pipeline(eng1, case["state"], {"q": q}).answers["q"]
        a2 = run_pipeline(eng2, case["state"], {"q": rev_q}).answers["q"]
        c1, c2 = a1.get("choice"), a2.get("choice")
        exp = case["expect"]["value"]
        if c1 != c2:
            flips += 1
        if c1 == exp:
            ok_orig += 1
        if c2 == exp:
            ok_rev += 1
        rows.append({"id": case["id"], "orig": c1, "rev": c2, "expected": exp, "flip": c1 != c2})
    out = {"backend": getattr(backend, "name", "?"), "model": getattr(backend, "model", ""),
           "n": len(bases), "flips": flips, "flip_rate": round(flips / max(len(bases), 1), 3),
           "acc_orig": round(ok_orig / max(len(bases), 1), 3),
           "acc_rev": round(ok_rev / max(len(bases), 1), 3), "rows": rows}
    os.makedirs("results", exist_ok=True)
    with open("results/reaction_bias.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def run_baseline(backend_name: str = "") -> Dict[str, Any]:
    """Logit-only vs free-text JSON generation on 12 base cases."""
    from .prompts import build_boolean_prompt as _bb, build_mc_prompt as _mc, build_score_prompt as _sc

    backend = get_backend(backend_name or os.getenv("FLASHBULLJEV_BACKEND", "fake"))
    if not hasattr(backend, "answer_json"):
        return {"skipped": "backend has no free-text generation (fake)"}
    picks = ["u01", "u04", "u07", "d01", "d05", "d09", "f01", "f05", "f09", "s01", "s05", "n01"]
    by_id = {c["id"]: c for c in _base_cases()}
    rows: List[Dict[str, Any]] = []
    for pid in picks:
        case = by_id[pid]
        q = dict(case["question"])
        t = str(q.get("type", "noul"))
        if t in ("noul", "boolean"):
            prompt = build_boolean_prompt(case["state"], str(q.get("instructions", "")))
            letters = ["A", "B"]
        elif t == "choice":
            opts = list((q.get("criteria", {}) or {}).keys())
            prompt = build_mc_prompt(case["state"], str(q.get("instructions", "")), opts)
            letters = ["A", "B", "C"][: len(opts)]
        elif t == "score":
            levels = list(q.get("criteria", []) or [])
            prompt = build_score_prompt(case["state"], str(q.get("instructions", "")), levels)
            letters = ["A", "B", "C"][: len(levels)]
        else:
            prompt = f"State: {case['state']}\nReply with JSON only: {{\"answer\": \"<value>\"}}"
            letters = []
        eng = FlashBullJevEngine(backend=backend)
        t0 = time.perf_counter()
        ans = run_pipeline(eng, case["state"], {"q": q}).answers["q"]
        ms_logit = (time.perf_counter() - t0) * 1000.0
        ok_logit = check_case(ans, dict(case["expect"]))
        text, ms_json = backend.answer_json(prompt + '\nReply with JSON only: {"answer": "A"}')
        letter = parse_json_letter(text, letters) if letters else None
        rows.append({"id": pid, "ok_logit": ok_logit, "ms_logit": round(ms_logit, 1),
                     "json_text": text[:120], "json_letter": letter, "ms_json": round(ms_json, 1)})
    n = len(rows)
    out = {"backend": getattr(backend, "name", "?"), "model": getattr(backend, "model", ""),
           "n": n, "acc_logit": round(sum(1 for r in rows if r["ok_logit"]) / n, 3),
           "avg_ms_logit": round(sum(r["ms_logit"] for r in rows) / n, 1),
           "avg_ms_json": round(sum(r["ms_json"] for r in rows) / n, 1), "rows": rows}
    os.makedirs("results", exist_ok=True)
    with open("results/reaction_baseline.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def print_reaction(out: Dict[str, Any]) -> None:
    """Print the rapid-fire table."""
    s = out["summary"]
    print(f"backend={s['backend']} model={s['model']} accuracy={s['correct']}/{s['n']}={s['accuracy']} avg={s['avg_ms']}ms contract={s['contract_ok']}/{s['n']}")
    bad = [r for r in out["rows"] if not r["ok"]][:12]
    for r in bad:
        print(f"KO  {r['id']} {r['ms']:>8}ms  <- {r['state'][:60]}")


def print_calibration(out: Dict[str, Any]) -> None:
    """Print the fit/eval calibration report."""
    print(f"backend={out['backend']} model={out['model']}")
    print(f"fit n={out['fit']['n']} T={out['fit']['temperature']} {out['fit']['report']}")
    print(f"eval T=1.0 : {out['eval_T1']}")
    print(f"eval T=fit : {out['eval_Tfit']}")
