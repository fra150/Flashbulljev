"""Benchmark: 3 concurrent 1-token logprob calls vs one base URL.

Shows whether the Ollama server parallelizes (OLLAMA_NUM_PARALLEL)
or serializes requests. Run: python scripts/bench_parallel.py [base_url]
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11434"
MODEL = "qwen2.5:1.5b"
PROMPTS = [
    "State: payout failing 3 days.\nQuestion: urgent?\nA) Yes\nB) No\nAnswer:",
    "State: app crashes on login.\nQuestion: team?\nA) billing\nB) technical\nC) sales\nAnswer:",
    "State: thanks, great fix!\nQuestion: frustration?\nA) Calm\nB) Frustrated\nC) Very angry\nAnswer:",
]


def one(prompt):
    body = {"model": MODEL, "prompt": prompt, "stream": False,
            "options": {"num_predict": 1, "temperature": 0},
            "logprobs": True, "top_logprobs": 5, "keep_alive": "30m"}
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/api/generate", json=body, timeout=120)
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000.0


with ThreadPoolExecutor(max_workers=3) as ex:
    t0 = time.perf_counter()
    lats = list(ex.map(one, PROMPTS))
    total = (time.perf_counter() - t0) * 1000.0
print(f"base={BASE} total={total:.0f}ms individual={[f'{x:.0f}' for x in lats]} sum={sum(lats):.0f}ms")
