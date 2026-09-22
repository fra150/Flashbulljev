# flashbulljev v0.4.0 — a real System One prototype

**Fast and stored: unstructured state in → typed probabilistic decisions out. Zero useful tokens.**

![flashbulljev](assets/imm001.jpg)

Inspired by:
- https://typesafe.ai/blog/introducing-system-one-models-and-jev (System One models / Jev)
- https://github.com/aayushch/laya (local-first AI command center)
- The logit-only System One pattern (zero useful tokens, Jev-compatible API)
- `Frammento del veloce` by dr. Bulla Francesco (g0/gx/gy + gf quiescence / certification / zero-cost memory)

## Test report (measured on this machine, Ollama Qwen 2B)

```
pytest:              50 passed
[fake]               miss ~52ms   hit ~0.08ms
[ollama+qwen2.5:1.5b] miss avg ~700ms (3 parallel questions, warm)  hit avg ~0.07ms
[ollama cold load]   first call ~3s (one-time model load); keep_alive=30m avoids reloads
live API:            /health, /v1/decisions, /v1/systemone, /v1/models — all OK, answers certified Q=0.99
battery (576 cases): Qwen 420/576 = 0.729 @ ~332ms/reaction, contract 576/576
battery (0.5b):      Qwen-0.5b 89/208 = 0.428 @ 246ms — faster but useless, 1.5b stays default
calibration (real):  fit n=330 T=3.0 (NLL 0.560/Brier 0.121/ECE 0.128)
                     held-out n=150: T=1 (1.147/0.160/0.211/acc 0.747) vs T=3 (0.544/0.125/0.139/acc 0.747)
position bias:       0 flips / 12, acc 0.917 original and reversed — no bias found
abstention:          tuned rule tau=(0.6, 0.3) → 1/12 vague declined, battery cost -0.010 (band-aid, model overconfident)
baseline:            logit-only 0.833 @ 337ms vs free-text JSON 0.727 @ 498ms (JSON needed repair, fenced markdown)
parallel:            NUM_PARALLEL=4 → 3 concurrent in 123ms vs 367ms default (~3x, needs Ollama restart)
```

The rule is confirmed:
- fast but NOT stored = you repay ~700ms on every call
- fast AND stored = ~0.07ms certified hit, ~10,000x faster (now also across restarts via SQLite)

## How the real logit-only flow works

1. Short System One prompt: `State + Question + A) B) C) + Answer with one letter:`
2. `POST /api/generate` to Ollama with `num_predict=1, temperature=0, logprobs=true, top_logprobs=adaptive, keep_alive=30m`
3. Read only the letter logprobs A/B/C, apply `exp` + renormalization → probabilities, ignore the token
4. `quiescence AND quality AND budget` via `certify_fragment`; store in `FragmentMemory` only when certified
5. Next identical call: `sha256(state+backend+questions)` → `recall()` in ~0.07ms (memory or SQLite file)

## What v0.4.0 adds (all 11 gaps closed or measured)

1. 208-case battery (52 curated x 4 label-preserving variants) — real calibration scale
2. Abstention probe: 12 vague cases — result 0/12, the known weakness is now measured
3. Position-bias probe: option reversal — 0 flips, no bias on this model
4. JSON baseline: free-text generation loses on accuracy (0.727 vs 0.833) and latency (498 vs 337ms)
5. Speed: cross-request KV reuse via `context` tested — NO gain on Ollama (369 vs 299ms, documented negative);
   shipped instead `keep_alive=30m` (no more 3s reloads) and adaptive `top_logprobs`
6. SQLite persistence: `FLASHBULLJEV_CACHE_DB` — cache hits survive restarts
7. API auth (`FLASHBULLJEV_API_KEY`, Bearer) + per-IP rate limit (`FLASHBULLJEV_RPS`), `/health` stays public
8. CI: coverage (`pytest-cov`) + `results/` artifact upload; Ollama live job scaffolded (disabled until a runner has Ollama)
9. `LICENSE` (MIT) now exists
10. Versions aligned at 0.4.0 (`pyproject`, engine, API, README)
11. Honest limits section (below) instead of hidden weaknesses

## Abstention (tuned, honest limits)

`python run.py react-tune ollama` grids (tau_abs, tau_margin) over clear vs vague
(max, margin) distributions. The engine enforces the winner
(`FLASHBULLJEV_ABSTAIN_TAU=0.6`, `FLASHBULLJEV_ABSTAIN_MARGIN=0.3`, env-overridable).
Effect: vague 0/12 → 1/12 declined at a battery cost of -0.010. Thresholds are a
band-aid — the model is overconfident at any temperature — the real fix is
calibration-aware abstention or a better model.

## Deploy (Windows production)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/serve_production.ps1  # env defaults + logs/
powershell -ExecutionPolicy Bypass -File scripts/backup_cache.ps1      # backups/, keeps newest 7
```

Measured serial capacity: single decision ~229ms → ~4.4/s, so the default
`FLASHBULLJEV_RPS=2` is the conservative half. For concurrent load set
`OLLAMA_NUM_PARALLEL=4` on the Ollama server (restart required) — measured
3 concurrent in 123ms vs 367ms default.

## Honest limits

- Qwen abstains 1/12 after tuning (was 0/12): thresholds help little, overconfidence is structural
- Held-out is now n=150 (was 56): ECE still noisy but NLL halved (1.15→0.54) is solid
- One resident model, localhost by default; concurrent Ollama requests serialize (~sum of latencies)
- Cross-request prefill sharing does not help on Ollama — multi-question cost stays ~N x single
- `qwen2.5:0.5b` is ~20% faster but drops to 0.428 accuracy: not viable

## Project layout

```
bulla-jev/
  assets/imm001.jpg
  src/flashbulljev/
    __init__.py      # public exports
    __main__.py      # CLI (see below)
    schemas.py       # boolean/choice/score/numeric questions
    prompts.py       # multiple-choice prompt builder, letter lists
    backends.py      # FakeBackend + OllamaBackend (logits + JSON baseline), get_backend
    calibration.py   # temperature scaling, nll/brier/ece, fit_temperature
    memory_gf.py     # verify_quiescence, certify_fragment, FragmentMemory (+SQLite)
    engine.py        # parallel decisions, FlashBullJevEngine (miss/hit, CACHE_DB)
    pipeline.py      # ingest -> route -> decide -> certify -> emit
    api.py           # FastAPI + Bearer auth + rate limit
    reaction.py      # 208-case battery, abstain/bias/baseline probes, calibration split
  tests/             # test_memory_gf/engine/api_pipeline/real/extra/reaction/persist/api_auth
  results/           # tracked battery + calibration + bias + baseline JSON
  scripts/           # probe_prefill.py, bench_parallel.py, score_baseline.py
  LICENSE
```

## Quickstart

```bash
pip install -r requirements.txt
python run.py demo
$env:FLASHBULLJEV_BACKEND="ollama"; $env:FLASHBULLJEV_MODEL="qwen2.5:1.5b"
python run.py demo
python run.py demo-real   # fake vs ollama side by side
python run.py react ollama       # 208-case rapid-fire battery
python run.py react-calib ollama # fit T + held-out eval
python run.py react-bias ollama  # option-reversal probe
python run.py react-base ollama  # logit-only vs JSON baseline
python run.py react-abstain ollama
python run.py bench
$env:FLASHBULLJEV_BACKEND="fake"; python -m pytest -q
```

Server:

```bash
$env:FLASHBULLJEV_BACKEND="ollama"
$env:FLASHBULLJEV_CACHE_DB="flashbulljev.db"  # persistent cache
$env:FLASHBULLJEV_API_KEY="changeme"          # Bearer auth
$env:FLASHBULLJEV_RPS="2"                     # per-IP rate limit (measured: serial ~4.4/s)
python run.py serve
```

## Environment variables

- `FLASHBULLJEV_BACKEND=fake|ollama` (default fake)
- `FLASHBULLJEV_MODEL=qwen2.5:1.5b` (default)
- `OLLAMA_URL=http://127.0.0.1:11434`
- `FLASHBULLJEV_TEMP=1.0` or `FLASHBULLJEV_CALIB=calibration-fit.json`
- `FLASHBULLJEV_CACHE_DB=` (empty = memory only)
- `FLASHBULLJEV_API_KEY=` (empty = auth off, localhost dev)
- `FLASHBULLJEV_RPS=0` (0 = no rate limit)
- `FLASHBULLJEV_ABSTAIN_TAU=0.6` + `FLASHBULLJEV_ABSTAIN_MARGIN=0.3` (tuned, env-overridable)

## Clean-code contract

- Small pure functions with type hints and English docstrings
- numpy arrays for state vectors, lists for keys/questions/steps
- 5-stage pipeline, each stage isolated and testable
- NEVER certified when quiescence is off (GF rule)
- Legacy Italian aliases kept for backward compatibility (`MemoriaGF`, `verifica_quiete`, ...)

## License

MIT — see LICENSE.
