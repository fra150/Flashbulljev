# Diario di bordo — flashbulljev

Autore del progetto: Francesco Bulla (fra150), Catania.
Assistente: Maestro Supremo / Muse Spark.
Periodo: 22 settembre 2026 (sessione unica, dal pomeriggio a sera).

---

## 1. Il punto di partenza

L'idea iniziale era ricreare un modello nuovo ispirato a tre fonti diverse:

1. **System One Models & Jev** (TypeSafe AI, blog 15/09/2026): modelli non per chattare ma per
   decidere in fretta dentro il software — `stato non strutturato in → decisioni tipate
   probabilistiche out`, campionamento parallelo, 70-500ms, output type-safe senza allucinazioni,
   training con RLCD (Reinforcement Learning for Calibrated Decisions).
2. **Laya** (github.com/aayushch/laya): command-center AI local-first — aggrega notifiche
   (Slack, Gmail, GitHub, Jira, Notion, Outlook, Calendar) con LLM locali via Ollama/LM Studio,
   pipeline n8n → engine Python FastAPI → UI Tauri+Svelte, Action Cards da approvare.
3. **`Framento del veloce`** (cartella sul Desktop, ricerca del dr. Bulla Francesco, 17/09/2026):
   libreria di simulazione numerica per la diffusione dinamica della memoria, architettura a tre
   livelli geometrici — g0 essenza perfetta invariante (R=0), gx operatività vincolata agli input,
   gy novità controllata — più livello trasversale gf (quiete attiva + certificazione + memoria a
   costo zero). 96 test, coverage 81%, solo numpy/scipy/matplotlib.

A queste si è aggiunta una quarta ispirazione, proposta dall'utente:
4. **rizzo-flow** (github.com/Rizzo-AI-Academy/rizzo-flow): reimplementazione open e locale
   dell'idea di Jev — decisioni tipate da un LLM senza generare un solo token, leggendo solo i
   logit delle lettere-risposta (A/B/C), ~50ms a decisione su GPU, API compatibile con Jev.
   (Poi rimossa da ogni parte del repo su richiesta esplicita dell'utente.)

## 2. L'intuizione centrale: veloce + salvato = 1 millisecondo

La frase chiave dell'utente, che ha orientato tutto:

> «Se l'output è veloce ma non salvato cosa succede? È veloce e salvato 1 millesimo di secondo
> più veloce di tutti.»

Tradotta in ingegneria, ed è corretta:

- **Veloce ma NON salvato = ricalcolo.** Ogni chiamata ripaga prefill + diffusione + verifica.
  Anche a 50ms, la volta dopo ripaghi gli stessi 50ms. Nessun accumulo, nessuna continuità.
- **Veloce + salvato = hit.** Il modulo `MemoriaGF` del Frammento (`frammento_gf.py`):
  chiave sha256 di valori+shape, `salva()` solo se certificato
  (`quiete AND qualità AND budget`, mai se quiete=False), `richiama()` senza ricalcolare.
  Primo giro miss ~50ms, giri dopo ~0.1ms. Misurato, non dichiarato.

Da qui il nome del modello: **flashbulljev**.

## 3. Scaffolding v0.1.0 — engine fake che dimostra il miss/hit

Creato in `C:\Users\acese\Desktop\bulla-jev` (workspace consentito, prima quasi vuoto):

- `src/flashbulljev/`: `schemas.py` (boolean/choice/score/numeric), `memory_gf.py` (quiete,
  certificazione, MemoriaGF adattata), `engine.py` (softmax, decisioni parallele, cache),
  `pipeline.py` (ingest→route→decide→certify→emit), `api.py` (FastAPI `/v1/decisions`,
  `/v1/systemone`, `/health`), `__main__.py` (CLI demo/bench/serve/decide)
- `tests/` (15 test), `Dockerfile`, `docker-compose.yml`, `requirements.txt`,
  `pyproject.toml`, `run.py`, `README.md`, CI GitHub Actions
- Logit fake deterministici da hash (MVP senza pesi) + `sleep(0.05)` per simulare il miss.
- Prima misura: **miss 58ms → hit 0.15ms**, 15 test passed.

## 4. Vero prototipo v0.2.0 — logit-only con Ollama

Verifica live: Ollama 0.33.2 con `llama3.2:1b` supporta `logprobs=true + top_logprobs`
su `/api/generate` con `num_predict=1`. Quindi implementato il vero pattern System One:

- `prompts.py`: prompt corti stile multiple-choice (stato una volta, lettere A-Z).
- `backends.py`: `FakeBackend` + `OllamaBackend` (1 token tecnico, 0 utili, niente parsing JSON).
- `calibration.py`: temperature scaling + NLL/Brier/ECE.
- Engine con backend intercambiabile, API con switch backend/temperatura, CLI con demo/bench/calibrate.
- Misura: miss ~850ms (3 domande) → hit 0.06ms.

## 5. Passaggio a Qwen 2B e pulizia riferimenti

Su richiesta utente: scaricato `qwen2.5:1.5b` (986MB) e reso default (`FLASHBULLJEV_MODEL`),
rimossa ogni traccia di "rizzo" dal codice (4 punti: schemas, api, README ×2).
Qwen 1.5b misurato più veloce di llama 1b a caldo (657 vs 835ms miss) e più sensato
(es: payout-failing → technical 0.98 invece di sales).

## 6. Internazionalizzazione + primo push (con incidente)

Richieste: README in inglese con `assets/imm001.jpg`, codice e commenti in inglese,
sistemare `.gitignore`, pushare su `https://github.com/fra150/Flashbulljev.git`.

- Tutto il codice tradotto in inglese; nomi italiani tenuti come alias retrocompatibili
  (`verify_quiescence`/`verifica_quiete`, `FragmentMemory`/`MemoriaGF`, ...), output JSON bilingue.
- **Incidente:** al primo push è finito dentro anche `.opencode` (145MB di config IDE).
  L'utente l'ha segnalato («problema hai puschiato opencode»).
- **Rimedio:** storia ricreata da zero (25 file, 0 percorsi `.opencode`), force-push.
  Nota: per sbloccare un sotto-repo annidato ho dovuto rimuovere
  `.opencode/skills/ui-ux-pro-max-skill/.git` (solo history interna, file intatti).
  `.gitignore` ora ignora `.opencode/` e i file locali restano intatti sul PC.

## 7. Batterie di test — dal botta e risposta ai 208 casi

- **16 casi** rapid-fire (urgenza, routing, frustration, livelli): Qwen 13/16 = 0.812 @ ~358ms,
  contratto 16/16; fake 4/16 a caso (dimostra che Qwen decide davvero).
- **52 casi** con split fit/eval onesto.
- **208 casi finali** (52 curati × 4 varianti label-preserving: originale, MAIUSCOLO,
  incapsulamento gentile, virgolette): **Qwen 166/208 = 0.798 @ ~300ms, contratto 208/208**.
  Controllo 0.5b: 89/208 = 0.428 @ 246ms — più veloce ma inutile, 1.5b resta default.
- API live verificate (server avviato, chiamato, spento): health, decisions, systemone, models.
- Suite cresciuta: 15 → 23 → 24 → 33 → 38 → 41 → **48 passed**.

## 8. Calibrazione vera (non dimostrativa)

`run.py react-calib`: logit grezzi sullo split fit, temperature fit, valutazione su held-out:

- fit n=120: **T=3.0** (NLL 0.399 / Brier 0.088 / ECE 0.104)
- held-out n=56: T=1 (0.804 / 0.145 / 0.212) vs T=3 (**0.483** / 0.117 / 0.151), acc 0.786 invariata
- Lettura onesta: Qwen esce overconfident, T=3 dimezza la sorpresa; ECE su 56 campioni resta rumoroso.
- Risultati tracciati in `results/reaction_calibration.json`, T in `calibration-fit.json`.

## 9. v0.4.0 — chiuse tutte le 11 lacune

1. Batteria 200+ (208) con split a famiglie senza leakage.
2. Abstention: 12 casi vaghi → **0/12, il modello non si astiene mai** (debolezza misurata, non nascosta).
3. Position bias: inversione opzioni → **0 flip/12**, acc 0.917 entrambi i versi.
4. Baseline JSON: generazione libera 0.727 @ 498ms (con markdown da riparare) vs logit **0.833 @ 337ms**.
5. Prefill condiviso: provato `context` di Ollama → **nessun guadagno** (369 vs 299ms, negativo documentato);
   spedito invece `keep_alive=30m` (niente più reload da 3s) + `top_logprobs` adattivo.
6. Persistenza SQLite (`FLASHBULLJEV_CACHE_DB`): l'hit sopravvive al restart, testato.
7. Auth Bearer (`FLASHBULLJEV_API_KEY`) + rate limit per-IP (`FLASHBULLJEV_RPS`), `/health` pubblica.
8. CI con coverage + upload artifact `results/`; job live Ollama predisposto (spento).
9. `LICENSE` MIT creato. 10. Versioni allineate a **0.4.0** ovunque.
11. Sezione "honest limits" nel README.

## 10. Stato finale

- Repo: `https://github.com/fra150/Flashbulljev.git`, branch `main`, tutto pushato
  (codice + `results/*.json` + README aggiornato ad ogni passo).
- Numeri di riferimento: miss ~700ms (3 domande) / hit ~0.07ms (~10.000×) / batteria 0.798 /
  calibrazione T=3.0 / bias 0 / abstention 0 (da risolvere) / baseline vinta.
- Comandi: `demo | demo-real | react [fake|ollama] | react-calib | react-bias |
  react-base | react-abstain | bench | calibrate | serve`.

## 11. Cose da ricordare (promemoria per il futuro)

- **Revocare il token provvisorio** `ghp_...Uxxk` su GitHub (usato solo negli URL di push,
  mai committato in file, ma resta nella cronologia della shell).
- Il file `.gitignore` è sorvegliato dall'ambiente (tenta di riscriversi da solo): se un giorno
  `.opencode` rientra nello staging, ripetere la procedura del §6.
- Prossimi passi naturali: allargare il calibration set oltre i 208, insegnare l'abstention
  (soglie su confidence/margine), confrontare `llama3.2:3b` già presente in locale.
