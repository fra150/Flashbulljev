"""Probe: does Ollama context-reuse give us shared-prefix prefill?

Compares FULL prompt vs PREFILL(state) + question-only with returned context.
Run: python scripts/probe_prefill.py
"""
import sys
import time

sys.path.insert(0, "src")

import requests

BASE = "http://127.0.0.1:11434"
MODEL = "qwen2.5:1.5b"
STATE = "Help! My payouts have been failing for 3 days, customers are angry."
Q = "Question: Does this convey urgency?\nA) Yes\nB) No\nAnswer with one letter:"


def call(prompt, context=None):
    body = {"model": MODEL, "prompt": prompt, "stream": False,
            "options": {"num_predict": 1, "temperature": 0},
            "logprobs": True, "top_logprobs": 5}
    if context is not None:
        body["context"] = context
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/api/generate", json=body, timeout=120)
    r.raise_for_status()
    return r.json(), (time.perf_counter() - t0) * 1000.0


full, ms_full = call(f"State: {STATE}\n{Q}")
print(f"FULL: ms={ms_full:.0f} prompt_eval={full.get('prompt_eval_count')} resp={full.get('response')!r}")

pre, ms_pre = call(f"State: {STATE}\n")
ctx = pre.get("context")
print(f"PREFILL: ms={ms_pre:.0f} prompt_eval={pre.get('prompt_eval_count')} ctx_len={len(ctx or [])}")

q2, ms_q = call(Q, context=ctx)
print(f"SHARED-Q: ms={ms_q:.0f} prompt_eval={q2.get('prompt_eval_count')} resp={q2.get('response')!r}")
print(f"TOTAL shared(first Q)={ms_pre + ms_q:.0f} vs full={ms_full:.0f}")
