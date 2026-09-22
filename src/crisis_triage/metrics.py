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
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score))
    sorted_scores = y_score[order]
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
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
