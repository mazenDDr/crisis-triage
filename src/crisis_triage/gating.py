"""Confidence gating: send confident answers automatically, the rest to a person.

Each message gets one decision and one confidence. Choice questions: the top class and its
probability. Yes/no questions per label (needs, sectors): the decision is correct only if every
label is right, and the confidence is that of the least confident label (P for a yes, 1 - P for
a no). The threshold that reaches a target accuracy is picked on dev and then applied to test,
where coverage and the accuracy actually reached are measured.
"""

from __future__ import annotations

import numpy as np


def choice_decisions(probs: np.ndarray, gold_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, correct) for one choice per message; probs is (n, classes)."""
    return probs.max(axis=1), probs.argmax(axis=1) == gold_idx


def multilabel_decisions(
    p: np.ndarray, y: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, correct) for yes/no per label; p and y are (n, labels)."""
    yes = p >= thresholds
    conf = np.where(yes, p, 1 - p).min(axis=1)
    return conf, (yes == y.astype(bool)).all(axis=1)


def threshold_for(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """The lowest confidence cut whose kept messages are at least `target` accurate, i.e. the
    largest coverage that meets the target (inf when even the most confident miss it).

    A cut keeps every message at or above it, so it can only sit between two different
    confidence values: saturated models give many answers exactly 1.0, and a group of ties
    is kept or dropped as a whole.
    """
    order = np.argsort(-conf, kind="mergesort")
    sorted_conf = conf[order]
    acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    group_end = np.append(sorted_conf[1:] != sorted_conf[:-1], True)  # last index of each tie
    ok = np.nonzero((acc >= target) & group_end)[0]
    return float(sorted_conf[ok.max()]) if len(ok) else float("inf")


def gate(conf: np.ndarray, correct: np.ndarray, cut: float) -> tuple[float, float]:
    """(coverage, accuracy of the covered messages) at a confidence cut."""
    kept = conf >= cut
    return float(kept.mean()), float(correct[kept].mean()) if kept.any() else float("nan")


def risk_coverage(conf: np.ndarray, correct: np.ndarray, points: int = 20) -> list[dict]:
    """Accuracy of the most confident fraction of messages, for coverage 5%, 10%, ... 100%."""
    order = np.argsort(-conf, kind="mergesort")
    out = []
    for c in np.linspace(1 / points, 1, points):
        k = max(1, int(round(c * len(order))))
        out.append({"coverage": round(float(c), 3), "accuracy": float(correct[order[:k]].mean())})
    return out


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (mean error over all coverages; lower is better)."""
    order = np.argsort(-conf, kind="mergesort")
    err = np.cumsum(~correct[order]) / np.arange(1, len(order) + 1)
    return float(err.mean())
