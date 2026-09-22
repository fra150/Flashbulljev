"""Real backends: Fake (deterministic) + Ollama logit-only (1 token + logprobs).

Ollama /api/generate with num_predict=1, logprobs=true, top_logprobs=20.
We read only the letter logits and ignore the generated token.
True System One: 0 useful tokens, 1 technical token, no JSON parsing.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import requests

from .prompts import letters_for

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("FLASHBULLJEV_MODEL", "qwen2.5:1.5b")


def softmax_list(logits: List[float], temperature: float = 1.0) -> List[float]:
    """Stable softmax over a list."""
    arr = np.array(list(logits), dtype=float) / max(float(temperature), 1e-6)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return [float(p) for p in exp / (np.sum(exp) + 1e-12)]


class FakeBackend:
    """Deterministic backend for tests/offline. No network."""

    name = "fake"

    def logits_for(self, prompt: str, n_options: int) -> Tuple[List[float], float]:
        """Hash-based fake logits, zero latency."""
        h = hashlib.sha256(prompt.encode()).digest()
        logits: List[float] = []
        for i in range(max(n_options, 2)):
            logits.append(float(h[(i * 3) % len(h)] + h[(i * 7 + 1) % len(h)] / 255.0))
        return logits, 0.0


class OllamaBackend:
    """Real Ollama backend: 1 token + top_logprobs -> letter probabilities."""

    name = "ollama"

    def __init__(self, model: str = "", base_url: str = "") -> None:
        self.model = model or OLLAMA_MODEL
        self.base_url = (base_url or OLLAMA_URL).rstrip("/")

    def logits_for(self, prompt: str, n_options: int) -> Tuple[List[float], float]:
        """Call Ollama and extract letter logprobs A... Returns (logits, latency_ms)."""
        letters: List[str] = letters_for(n_options)
        t0 = time.perf_counter()
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 1, "temperature": 0},
            "logprobs": True,
            "top_logprobs": 20,
        }
        r = requests.post(f"{self.base_url}/api/generate", json=body, timeout=120)
        r.raise_for_status()
        data: Dict[str, Any] = r.json()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        top: List[Dict[str, Any]] = []
        try:
            top = list(data.get("logprobs", [{}])[0].get("top_logprobs", []))
        except (IndexError, AttributeError):
            top = []
        # token -> logprob map, stripped tokens
        m: Dict[str, float] = {}
        for e in top:
            tok = str(e.get("token", "")).strip()
            if tok and tok not in m:
                try:
                    m[tok] = float(e.get("logprob", -20.0))
                except (TypeError, ValueError):
                    m[tok] = -20.0
        logits: List[float] = []
        for letter in letters:
            logits.append(float(m.get(letter, -20.0)))
        return logits, dt_ms


def get_backend(name: str = "") -> Any:
    """Backend factory from env FLASHBULLJEV_BACKEND=fake|ollama."""
    b = (name or os.getenv("FLASHBULLJEV_BACKEND", "fake")).lower()
    if b == "ollama":
        return OllamaBackend()
    return FakeBackend()
