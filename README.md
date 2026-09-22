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
pytest:              41 passed
[fake]               miss ~52ms   hit ~0.08ms
[ollama+qwen2.5:1.5b] miss avg ~715ms (3 parallel questions, warm)  hit avg ~0.07ms
[ollama cold load]   first call ~3.1s (one-time model load), then warm timings above
live API:            /health, /v1/decisions, /v1/systemone, /v1/models — all OK, answers certified Q=0.99
reaction battery:    Qwen 44/52 = 0.846 @ ~313ms/reaction, contract 52/52 (fake baseline random)
calibration (real):  fit n=30 T=3.0 NLL 0.385; held-out eval n=14: T=1 NLL 0.662 ECE 0.178 / T=3 NLL 0.409 ECE 0.217, acc 0.857
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
  results/               # tracked battery + calibration JSON (see below)
    reaction_ollama.json reaction_calibration.json
  calibration-fit.json   # fitted temperature (real fit, Qwen 2B, n=30)
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
python run.py react ollama  # 52-case rapid-fire Q&A battery (also: react fake)
python run.py react-calib ollama  # fit T on 36 cases, evaluate on held-out 16
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

`src/flashbulljev/reaction.py` fires 52 labeled one-question probes — urgency (12),
sentiment (8), department routing (12), frustration score (12), fill-level reading (8) —
and scores reaction vs expectation plus output-contract checks
(type/status/confidence ranges, probabilities summing to 1).
Every 3rd case per group is held out for eval; the rest is the fit set, so the
battery doubles as a real calibration set (`run.py react-calib`).

```
Qwen 2B: 44/52 = 0.846 @ ~313ms/reaction, contract 52/52
fake:     random baseline @ ~51ms/reaction, contract 52/52
```

## Calibration (real fit, held-out eval)

`python run.py react-calib ollama` collects raw letter logits on the fit split
(n=30 classifiable), fits the temperature, and evaluates on the held-out split (n=14):

```
fit:      T=3.0  NLL 0.385  Brier 0.085  ECE 0.098
held-out: T=1.0  NLL 0.662  Brier 0.099  ECE 0.178  acc 0.857
held-out: T=3.0  NLL 0.409  Brier 0.092  ECE 0.217  acc 0.857
```

Honest read: T=3.0 halves held-out surprise (NLL 0.66→0.41) — Qwen is overconfident
out of the box — while accuracy is unchanged and ECE on 14 samples is noise.
Full numbers in `results/reaction_calibration.json`; fitted T in `calibration-fit.json`.

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
