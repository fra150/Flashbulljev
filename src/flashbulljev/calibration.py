"""Temperature scaling calibration + metrics. Pure functions, lists."""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np


def apply_temperature(logits: List[float], temperature: float) -> List[float]:
    """Scale logits by temperature."""
    arr = np.array(list(logits), dtype=float) / max(float(temperature), 1e-6)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return [float(p) for p in exp / (np.sum(exp) + 1e-12)]


def nll(probs_list: List[List[float]], labels: List[int]) -> float:
    """Mean negative log-likelihood."""
    vals: List[float] = []
    for probs, y in zip(probs_list, labels):
        vals.append(-math.log(max(probs[y], 1e-12)))
    return float(sum(vals) / max(len(vals), 1))


def brier(probs_list: List[List[float]], labels: List[int]) -> float:
    """Mean Brier score."""
    vals: List[float] = []
    for probs, y in zip(probs_list, labels):
        s = 0.0
        for i, p in enumerate(list(probs)):
            t = 1.0 if i == y else 0.0
            s += (p - t) ** 2
        vals.append(s / len(probs))
    return float(sum(vals) / max(len(vals), 1))


def ece(probs_list: List[List[float]], labels: List[int], n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins: List[List[float]] = [[] for _ in range(n_bins)]
    bin_corr: List[List[int]] = [[] for _ in range(n_bins)]
    for probs, y in zip(probs_list, labels):
        conf = max(list(probs))
        pred = int(np.argmax(np.array(probs)))
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append(conf)
        bin_corr[b].append(1 if pred == y else 0)
    total = max(len(labels), 1)
    err = 0.0
    for b in range(n_bins):
        if bins[b]:
            acc = sum(bin_corr[b]) / len(bins[b])
            conf_m = sum(bins[b]) / len(bins[b])
            err += abs(acc - conf_m) * len(bins[b]) / total
    return float(err)


def fit_temperature(logits_list: List[List[float]], labels: List[int]) -> Tuple[float, Dict[str, Any]]:
    """Grid-search the temperature that minimizes NLL. Returns (T, report)."""
    candidates: List[float] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    best_t = 1.0
    best_nll = float("inf")
    probs_best: List[List[float]] = []
    for temp in candidates:
        probs: List[List[float]] = [apply_temperature(list(lg), temp) for lg in logits_list]
        v = nll(probs, labels)
        if v < best_nll:
            best_nll = v
            best_t = temp
            probs_best = probs
    report: Dict[str, Any] = {
        "temperature": best_t,
        "nll": best_nll,
        "brier": brier(probs_best, labels),
        "ece": ece(probs_best, labels),
        "n": len(labels),
    }
    return best_t, report


def save_fit(path: str, temperature: float, report: Dict[str, Any]) -> None:
    """Save a calibration fit."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"temperature": temperature, "report": report}, f, indent=2)


def load_temperature(path: str) -> float:
    """Load a temperature value, default 1.0."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("temperature", 1.0))
    except (OSError, ValueError):
        return 1.0
