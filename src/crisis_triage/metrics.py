"""Scores with no dependencies beyond numpy: macro-F1, ROC AUC and a paired bootstrap."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def f1_per_class(y_true, y_pred, labels) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {}
    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        out[c] = float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 0.0
    return out


def macro_f1(y_true, y_pred, labels=None) -> float:
    """Mean F1 over `labels` (default: classes present in y_true)."""
    labels = sorted(set(np.asarray(y_true))) if labels is None else labels
    return float(np.mean(list(f1_per_class(y_true, y_pred, labels).values())))


def binary_f1(y_true, y_score, threshold: float = 0.5) -> float:
    y_true, pred = np.asarray(y_true, bool), np.asarray(y_score) >= threshold
    tp, fp, fn = np.sum(pred & y_true), np.sum(pred & ~y_true), np.sum(~pred & y_true)
    return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 0.0


def auc(y_true, y_score) -> float:
    """ROC AUC by ranks (ties get the average rank). NaN when only one class is present."""
    y_true, y_score = np.asarray(y_true, bool), np.asarray(y_score, float)
    n_pos, n_neg = y_true.sum(), (~y_true).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Average rank per distinct score: tied scores share the mean of their positions.
    _, inverse, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def paired_bootstrap(
    metric: Callable[[np.ndarray], float],
    n: int,
    reps: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Point value and 95% interval of `metric(idx)` over resampled row indices.

    For a paired difference, `metric` computes (A - B) on the same resampled rows.
    """
    rng = np.random.default_rng(seed)
    full = metric(np.arange(n))
    boots = [metric(rng.integers(0, n, n)) for _ in range(reps)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"value": round(full, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def reliability(conf, correct, bins: int = 10) -> list[dict]:
    """Equal-width confidence bins: mean confidence, accuracy and count in each."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return [
        {
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": int((idx == b).sum()),
            "conf": float(conf[idx == b].mean()) if (idx == b).any() else None,
            "acc": float(correct[idx == b].mean()) if (idx == b).any() else None,
        }
        for b in range(bins)
    ]


def ece(conf, correct, bins: int = 10) -> float:
    """Expected calibration error: count-weighted |accuracy - confidence| over bins."""
    n = len(conf)
    return float(
        sum(
            b["n"] / n * abs(b["acc"] - b["conf"])
            for b in reliability(conf, correct, bins)
            if b["n"]
        )
    )


def brier(probs, onehot) -> float:
    """Mean squared error of probabilities: (n, k) for k classes, or (n,) for yes/no."""
    probs, onehot = np.asarray(probs, float), np.asarray(onehot, float)
    err = (probs - onehot) ** 2
    return float(err.sum(axis=1).mean() if err.ndim == 2 else err.mean())


def best_threshold(y_true, y_score) -> float:
    """The score threshold with the highest F1 (chosen on dev, applied to test)."""
    y_score = np.asarray(y_score, float)
    candidates = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 99)))
    return float(max(candidates, key=lambda t: binary_f1(y_true, y_score, t)))
