"""Scoring helpers shared by the tuning and evaluation scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crisis_triage.data import PROCESSED
from crisis_triage.metrics import auc, binary_f1

N_DEV = 1000


def dev_sample(track: str) -> pd.DataFrame:
    """The fixed dev sample every tuning run uses (seed 0)."""
    df = pd.read_parquet(PROCESSED / f"{track}.parquet")
    dev = df[df["split"] == "dev"]
    return dev.sample(min(N_DEV, len(dev)), random_state=0).reset_index(drop=True)


def multilabel_scores(answers: list[dict], variant: str, labels: list[str]) -> np.ndarray:
    """(n, labels) scores: P(true) for noul variants, P(option) for the choice variant."""
    if variant.startswith("noul"):
        return np.array([[a[f"{variant}/{c}"]["p"] for c in labels] for a in answers])
    return np.array([[a[f"{variant}/need"]["probs"][c] for c in labels] for a in answers])


def score_multilabel(y: np.ndarray, s: np.ndarray, labels: list[str]) -> dict:
    per = {
        c: {"auc": auc(y[:, j], s[:, j]), "f1@0.5": binary_f1(y[:, j], s[:, j])}
        for j, c in enumerate(labels)
    }
    return {
        "mean_auc": float(np.nanmean([p["auc"] for p in per.values()])),
        "macro_f1@0.5": float(np.mean([p["f1@0.5"] for p in per.values()])),
        "per_label": per,
    }


def top(probs: dict) -> str:
    return max(probs, key=probs.get)


def mean_auc_on(y, s):
    return lambda i: float(np.nanmean([auc(y[i, j], s[i, j]) for j in range(y.shape[1])]))
