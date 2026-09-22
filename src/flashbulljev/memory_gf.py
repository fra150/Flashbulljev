"""FragmentMemory (GF layer) - fast and stored = ~1ms.

Adapted from the "Frammento del veloce" research (g0/gx/gy + gf).
Rule: fast but NOT stored -> recompute every time.
Fast AND stored -> certified cache hit, near-zero cost.

Pure functions + numpy arrays + lists, clean code.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any, Dict, List, Tuple

import numpy as np

EPS = 1e-12


def _round(a: np.ndarray) -> np.ndarray:
    """Round to 1e-6 for a stable hash."""
    return np.round(np.asarray(a, dtype=float), 6)


def state_to_vector(state: Any) -> np.ndarray:
    """Convert a str|dict|list state to a float vector. Pure function."""
    if isinstance(state, np.ndarray):
        return state.astype(float).ravel()
    if isinstance(state, (list, tuple)):
        vals: List[float] = []
        for x in list(state):
            try:
                vals.append(float(x))
            except (TypeError, ValueError):
                vals.append(float(len(str(x)) % 97))
        return np.array(vals, dtype=float)
    if isinstance(state, dict):
        vals2: List[float] = []
        keys: List[str] = sorted(state.keys())
        for k in keys:
            v = state[k]
            try:
                vals2.append(float(v))
            except (TypeError, ValueError):
                s = str(v)
                vals2.append(float(sum(ord(c) for c in s[:32]) % 997))
        return np.array(vals2, dtype=float)
    s2 = str(state)
    arr: List[float] = [float(ord(c) % 128) for c in s2[:256]]
    if not arr:
        return np.zeros(4, dtype=float)
    return np.array(arr, dtype=float)


def verify_quiescence(
    fo: np.ndarray,
    fx: np.ndarray,
    essence: np.ndarray,
    eps: float = 0.60,
    delta: float = 0.90,
    fy: np.ndarray | None = None,
    novelty_budget_rel: float = 0.25,
) -> Dict[str, Any]:
    """Check that the operational state is quiescent vs the essence.

    Returns a dict with the active flag plus error metrics.
    """
    fo_a = np.asarray(fo, dtype=float)
    fx_a = np.asarray(fx, dtype=float)
    es_a = np.asarray(essence, dtype=float)
    n_es = float(np.linalg.norm(es_a))
    n_fo = float(np.linalg.norm(fo_a))
    err_fo_es = float(np.linalg.norm(fo_a - es_a) / (n_es + EPS))
    err_fx_fo = float(np.linalg.norm(fx_a - fo_a) / (n_fo + EPS))
    active = bool((err_fo_es < delta) and (err_fx_fo < eps))
    reasons: List[str] = []
    if err_fo_es >= delta:
        reasons.append(f"essence drift {err_fo_es:.4f}>={delta}")
    if err_fx_fo >= eps:
        reasons.append(f"not aligned {err_fx_fo:.4f}>={eps}")
    if fy is None:
        nov_rel = 0.0
    else:
        nov_rel = float(np.linalg.norm(np.asarray(fy, dtype=float)) / (n_es + EPS))
        if nov_rel > novelty_budget_rel:
            active = False
            reasons.append(f"excess novelty {nov_rel:.4f}>{novelty_budget_rel}")
    reason = (
        f"quiescence active err_fx_fo={err_fx_fo:.4f} nov={nov_rel:.4f}"
        if active
        else "quiescence inactive: " + ("; ".join(reasons) if reasons else "thresholds")
    )
    return {
        "active": active,
        "err_fx_fo": err_fx_fo,
        "err_fo_ess": err_fo_es,
        "novelty_rel": nov_rel,
        "reason": reason,
        # legacy aliases (Italian)
        "attivo": active,
        "novita_rel": nov_rel,
        "motivo": reason,
    }


# legacy alias
def verifica_quiete(fo, fx, essenza, eps=0.60, delta=0.90, fy=None, budget_novita_rel=0.25):
    """Legacy alias for verify_quiescence."""
    return verify_quiescence(fo, fx, essenza, eps=eps, delta=delta, fy=fy, novelty_budget_rel=budget_novita_rel)


def certify_fragment(
    fo: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    essence: np.ndarray,
    quality_threshold: float = 0.40,
    novelty_budget_rel: float = 0.30,
) -> Dict[str, Any]:
    """Certify only if quiescence AND quality AND budget hold. Never if quiescence is off."""
    quiescence = verify_quiescence(fo, fx, essence, fy=fy, novelty_budget_rel=novelty_budget_rel)
    fx_a = np.asarray(fx, dtype=float)
    es_a = np.asarray(essence, dtype=float)
    n_es = float(np.linalg.norm(es_a))
    quality = float(1.0 - np.linalg.norm(fx_a - es_a) / (n_es + EPS))
    nov_rel = float(quiescence["novelty_rel"])
    certified = bool(quiescence["active"] and quality >= quality_threshold and nov_rel <= novelty_budget_rel)
    if not quiescence["active"]:
        reason = f"not certified: {quiescence['reason']}"
    elif quality < quality_threshold:
        reason = f"not certified: quality={quality:.4f}<{quality_threshold}"
    elif nov_rel > novelty_budget_rel:
        reason = f"not certified: novelty {nov_rel:.4f}>{novelty_budget_rel}"
    else:
        reason = f"certified Q={quality:.3f} nov={nov_rel:.4f}"
    return {
        "certified": certified,
        "quality": quality,
        "quiescence": quiescence,
        "reason": reason,
        "novelty_rel": nov_rel,
        # legacy aliases (Italian)
        "certificato": certified,
        "qualita": quality,
        "quiete": quiescence,
        "motivo": reason,
        "novita_rel": nov_rel,
    }


# legacy alias
def certifica_frammento(fo, fx, fy, essenza, soglia_qualita=0.40, budget_novita_rel=0.30):
    """Legacy alias for certify_fragment."""
    return certify_fragment(fo, fx, fy, essenza, quality_threshold=soglia_qualita, novelty_budget_rel=budget_novita_rel)


class FragmentMemory:
    """Cache of certified fragments (recall at ~zero cost).

    Key: sha256 of values + shape. hit -> ~1ms, miss -> recompute.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._saves = 0
        self._db_path: str = ""

    def key(self, state: Any, question_keys: List[str]) -> str:
        """Hash of state + question keys."""
        vec = _round(state_to_vector(state))
        h = hashlib.sha256()
        h.update(str(vec.shape).encode())
        h.update(vec.tobytes())
        h.update("|".join(sorted(question_keys)).encode())
        return h.hexdigest()

    # legacy alias
    def chiave(self, state: Any, questions_keys: List[str]) -> str:
        """Legacy alias for key."""
        return self.key(state, questions_keys)

    def save(self, key: str, payload: Dict[str, Any]) -> None:
        """Store a copy of the payload (write-through to disk when attached)."""
        self._cache[str(key)] = dict(payload)
        self._saves += 1
        if self._db_path:
            try:
                con = sqlite3.connect(self._db_path)
                con.execute("CREATE TABLE IF NOT EXISTS fragments (k TEXT PRIMARY KEY, payload TEXT)")
                con.execute("INSERT OR REPLACE INTO fragments (k, payload) VALUES (?, ?)",
                            (str(key), json.dumps(dict(payload))))
                con.commit()
                con.close()
            except OSError:
                pass

    # legacy alias
    def salva(self, chiave: str, payload: Dict[str, Any]) -> None:
        """Legacy alias for save."""
        return self.save(chiave, payload)

    def recall(self, key: str) -> Tuple[bool, Dict[str, Any] | None]:
        """Recall without recomputing."""
        k = str(key)
        if k in self._cache:
            self._hits += 1
            return True, self._cache[k]
        self._misses += 1
        return False, None

    # legacy alias
    def richiama(self, chiave: str) -> Tuple[bool, Dict[str, Any] | None]:
        """Legacy alias for recall."""
        return self.recall(chiave)

    def stats(self) -> Dict[str, int]:
        """Cache counters: hits, misses, saves, items."""
        return {
            "hits": int(self._hits),
            "misses": int(self._misses),
            "saves": int(self._saves),
            "items": int(len(self._cache)),
            # legacy aliases (Italian)
            "salvataggi": int(self._saves),
            "elementi": int(len(self._cache)),
        }

    def keys_list(self) -> List[str]:
        """List of stored keys."""
        return list(self._cache.keys())

    def attach_db(self, path: str) -> int:
        """Attach a SQLite file: load existing entries, write through on save. Returns loaded count."""
        self._db_path = str(path)
        loaded = 0
        try:
            con = sqlite3.connect(self._db_path)
            con.execute("CREATE TABLE IF NOT EXISTS fragments (k TEXT PRIMARY KEY, payload TEXT)")
            for k, p in con.execute("SELECT k, payload FROM fragments"):
                try:
                    self._cache[str(k)] = json.loads(str(p))
                    loaded += 1
                except ValueError:
                    continue
            con.close()
        except OSError:
            pass
        return loaded


# legacy alias
MemoriaGF = FragmentMemory


def timed_call(fn, *args, **kwargs) -> Tuple[Any, float]:
    """Run fn and measure milliseconds. Clean helper."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return out, dt_ms
