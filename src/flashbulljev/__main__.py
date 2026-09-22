"""flashbulljev CLI: fake + ollama + calibration. International English version."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

from .backends import get_backend
from .calibration import fit_temperature, save_fit
from .engine import FlashBullJevEngine
from .pipeline import run_pipeline
from .reaction import print_calibration, print_reaction, run_reaction, run_reaction_calibration


def demo_questions() -> Dict[str, Any]:
    """Demo questions using lists."""
    return {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages", "sales": None},
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    }


def make_engine_from_env() -> FlashBullJevEngine:
    """Build the engine from the FLASHBULLJEV_BACKEND env var."""
    backend = get_backend(os.getenv("FLASHBULLJEV_BACKEND", "fake"))
    return FlashBullJevEngine(backend=backend)


def cmd_demo() -> None:
    """Show miss vs hit with the current backend."""
    eng = make_engine_from_env()
    state = "Help! My payouts have been failing for 3 days."
    qs = demo_questions()
    print(f"backend={getattr(eng.backend, 'name', '?')} model={getattr(eng.backend, 'model', '')}")
    r1 = run_pipeline(eng, state, qs)
    r2 = run_pipeline(eng, state, qs)
    print(f"miss: {r1.latency_ms:.2f}ms hit={r1.cache_hit} steps={r1.steps}")
    print(f"hit : {r2.latency_ms:.2f}ms hit={r2.cache_hit} steps={r2.steps}")
    print(f"stats: {eng.memory.stats()}")
    print("answers:", json.dumps(r2.answers, indent=2)[:2000])


def cmd_demo_real() -> None:
    """Real demo: compare fake vs ollama backends."""
    for bname in ["fake", "ollama"]:
        try:
            eng = FlashBullJevEngine(backend=get_backend(bname))
            qs = demo_questions()
            state = "Help! My payouts have been failing for 3 days."
            r1 = run_pipeline(eng, state, qs)
            r2 = run_pipeline(eng, state, qs)
            print(f"[{bname}] miss {r1.latency_ms:.1f}ms hit {r2.latency_ms:.2f}ms answers={list(r2.answers.keys())}")
        except (OSError, ValueError, RuntimeError) as e:
            print(f"[{bname}] skip: {e}")


def cmd_decide(path: str) -> None:
    """Decide from a JSON file {state, questions}."""
    eng = make_engine_from_env()
    with open(path, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    res = run_pipeline(eng, data.get("state"), dict(data.get("questions", {})))
    print(json.dumps({"answers": res.answers, "cache_hit": res.cache_hit, "latency_ms": res.latency_ms}, indent=2))


def cmd_react(which: str = "") -> None:
    """Rapid-fire reaction battery against fake or live backend."""
    out = run_reaction(which or os.getenv("FLASHBULLJEV_BACKEND", "fake"))
    print_reaction(out)


def cmd_react_calib(which: str = "") -> None:
    """Fit temperature on fit split, evaluate on held-out split."""
    out = run_reaction_calibration(which or os.getenv("FLASHBULLJEV_BACKEND", "fake"))
    print_calibration(out)


def cmd_bench() -> None:
    """Benchmark 5 states x2 rounds."""
    eng = make_engine_from_env()
    states: List[str] = [f"ticket {i} payout failing urgent" for i in range(5)]
    qs = demo_questions()
    miss_times: List[float] = []
    hit_times: List[float] = []
    for s in states:
        t0 = time.perf_counter()
        run_pipeline(eng, s, qs)
        miss_times.append((time.perf_counter() - t0) * 1000.0)
    for s in states:
        t0 = time.perf_counter()
        run_pipeline(eng, s, qs)
        hit_times.append((time.perf_counter() - t0) * 1000.0)
    print(f"miss avg {sum(miss_times)/len(miss_times):.2f}ms list={[round(x,1) for x in miss_times]}")
    print(f"hit  avg {sum(hit_times)/len(hit_times):.2f}ms list={[round(x,2) for x in hit_times]}")


def cmd_calibrate() -> None:
    """Fit the temperature on a small labeled demo set."""
    from .backends import FakeBackend

    # demo set: 6 boolean examples urgent? label 1=yes 0=no
    dataset: List[Dict[str, Any]] = [
        {"state": "payout failing 3 days help!", "label": 1},
        {"state": "thanks all good", "label": 0},
        {"state": "URGENT outage now", "label": 1},
        {"state": "hello how are you", "label": 0},
        {"state": "refund needed immediately", "label": 1},
        {"state": "ok see you tomorrow", "label": 0},
    ]
    be = get_backend(os.getenv("FLASHBULLJEV_BACKEND", "fake"))
    if isinstance(be, FakeBackend):
        print("fake calibration: demonstration only")
    from .prompts import build_boolean_prompt

    logits_list: List[List[float]] = []
    labels: List[int] = []
    for ex in dataset:
        prompt = build_boolean_prompt(ex["state"], "Does this convey urgency?")
        lg, _ = be.logits_for(prompt, 2)
        logits_list.append(list(lg))
        labels.append(int(ex["label"]))
    temp, report = fit_temperature(logits_list, labels)
    save_fit("calibration-fit.json", temp, report)
    print(f"T={temp} report={report}")


def main(argv: List[str] | None = None) -> None:
    """CLI entry point."""
    args: List[str] = list(argv or sys.argv[1:] or ["demo"])
    cmd = args[0].lower()
    if cmd == "demo":
        cmd_demo()
    elif cmd == "demo-real":
        cmd_demo_real()
    elif cmd == "decide" and len(args) > 1:
        cmd_decide(args[1])
    elif cmd == "react":
        cmd_react(args[1] if len(args) > 1 else "")
    elif cmd == "react-calib":
        cmd_react_calib(args[1] if len(args) > 1 else "")
    elif cmd == "bench":
        cmd_bench()
    elif cmd == "calibrate":
        cmd_calibrate()
    elif cmd == "serve":
        import uvicorn

        uvicorn.run("flashbulljev.api:app", host="127.0.0.1", port=8018, reload=False)
    else:
        print("usage: python -m flashbulljev [demo|demo-real|react [fake|ollama]|react-calib [fake|ollama]|decide file.json|bench|calibrate|serve]")


if __name__ == "__main__":
    main()
