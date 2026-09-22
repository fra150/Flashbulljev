# flashbulljev v0.3.0 — a real System One prototype

**Fast and stored: unstructured state in → typed probabilistic decisions out. Zero useful tokens.**

![flashbulljev](assets/imm001.jpg)

Inspired by:
- https://typesafe.ai/blog/introducing-system-one-models-and-jev (System One models / Jev)
- https://github.com/aayushch/laya (local-first AI command center)
- The logit-only System One pattern (zero useful tokens, Jev-compatible API)
- `Frammento del veloce` by dr. Bulla Francesco (g0/gx/gy + gf quiescence / certification / zero-cost memory)

## Test report (measured on this machine, Ollama Qwen 2B)

```
pytest:              38 passed
[fake]               miss ~52ms   hit ~0.08ms
[ollama+qwen2.5:1.5b] miss avg ~715ms (3 parallel questions, warm)  hit avg ~0.07ms
[ollama cold load]   first call ~3.1s (one-time model load), then warm timings above
live API:            /health, /v1/decisions, /v1/systemone, /v1/models — all OK, answers certified Q=0.99
reaction battery:    Qwen 13/16 = 0.812 @ ~358ms/reaction, contract 16/16 (fake baseline 4/16 random)
```

The rule is confirmed:
- fast but NOT stored = you repay ~700ms on every call
- fast AND stored = ~0.07ms certified hit, ~10,000x faster

## How the real logit-only flow works

1. Short System One prompt: `State + Question + A) B) C) + Answer with one letter:`
2. `POST /api/generate` to Ollama with `num_predict=1, temperature=0, logprobs=true, top_logprobs=20`
3. Read only the letter logprobs A/B/C, apply `exp` + renormalization → probabilities, ignore the token
4. `quiescence AND quality AND budget` via `certify_fragment`; store in `FragmentMemory` only when certified
5. Next identical call: `sha256(state+backend+questions)` → `recall()` in ~0.07ms

## Project layout (clean code, functions, arrays, lists)

```
bulla-jev/
  assets/imm001.jpg
  src/flashbulljev/
    __init__.py      # public exports
    __main__.py      # CLI demo|demo-real|bench|calibrate|serve|decide
    schemas.py       # boolean/choice/score/numeric questions
    prompts.py       # multiple-choice prompt builder, letter lists
    backends.py      # FakeBackend + OllamaBackend.logits_for, get_backend
    calibration.py   # temperature scaling, nll/brier/ece, fit_temperature
    memory_gf.py     # verify_quiescence, certify_fragment, FragmentMemory
    engine.py        # parallel decisions, FlashBullJevEngine (miss/hit)
    pipeline.py      # ingest -> route -> decide -> certify -> emit
    api.py           # FastAPI /v1/decisions /v1/systemone /health
  tests/
    test_memory_gf.py test_engine.py test_api_pipeline.py test_real.py test_extra.py test_reaction.py
  Dockerfile
  docker-compose.yml
  requirements.txt
  run.py
```

## Quickstart

```bash
pip install -r requirements.txt
# fast fake backend for development
python run.py demo
# real backend with Ollama Qwen 2B
$env:FLASHBULLJEV_BACKEND="ollama"; $env:FLASHBULLJEV_MODEL="qwen2.5:1.5b"
python run.py demo
python run.py demo-real   # fake vs ollama side by side
python run.py react ollama  # 16-case rapid-fire Q&A battery (also: react fake)
python run.py bench       # 5 states x2 rounds (miss vs hit)
python run.py calibrate   # fit temperature on the demo set
$env:FLASHBULLJEV_BACKEND="fake"; python -m pytest -q
```

Server:

```bash
$env:FLASHBULLJEV_BACKEND="ollama"; python run.py serve
# API examples
# POST /v1/decisions {"state": ..., "questions": ..., "backend": "ollama"}
# POST /v1/systemone (Jev-compatible)
# GET  /health, GET /v1/models
```

## Reaction battery (rapid Q&A)

`src/flashbulljev/reaction.py` fires 16 labeled one-question probes — urgency yes/no,
department routing, frustration score, fill-level reading — and scores reaction vs expectation
plus output-contract checks (type/status/confidence ranges, probabilities summing to 1).

```
Qwen 2B: 13/16 = 0.812 @ ~358ms/reaction, contract 16/16
fake:     4/16 = 0.25  @ ~51ms/reaction,  contract 16/16  (hash-random baseline)
```

Honest misses: invoice-reminder flagged urgent (debatable), one thanks scored 0.78
near-tie, one all-caps rage scored 1.01 instead of 2 (hedged). Two of three are
near-ties, not confident errors.

## Calibration note

`python run.py calibrate` fits the temperature on a 6-example demo set (T=3.0 on Qwen 2B, NLL 2.48 / ECE 0.89).
That only proves the tooling works — real calibration needs a labeled set from your own domain,
plus a separate calibration split and a held-out test, exactly like the big labs do.

## Docker

```bash
docker build -t flashbulljev .
docker run -p 8018:8018 -e FLASHBULLJEV_BACKEND=fake flashbulljev
docker compose up --build
# for the real Ollama backend: OLLAMA_URL=http://host.docker.internal:11434
```

## Environment variables

- `FLASHBULLJEV_BACKEND=fake|ollama` (default fake)
- `FLASHBULLJEV_MODEL=qwen2.5:1.5b` (default)
- `OLLAMA_URL=http://127.0.0.1:11434`
- `FLASHBULLJEV_TEMP=1.0` or `FLASHBULLJEV_CALIB=calibration-fit.json`

## Clean-code contract

- Small pure functions with type hints and English docstrings
- numpy arrays for state vectors, lists for keys/questions/steps
- 5-stage pipeline, each stage isolated and testable
- NEVER certified when quiescence is off (GF rule)
- Legacy Italian aliases kept for backward compatibility (`MemoriaGF`, `verifica_quiete`, ...)

## License

MIT — see LICENSE.
