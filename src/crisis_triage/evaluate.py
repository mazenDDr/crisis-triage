"""Score any system's answers the same way: Laya, the LLM baselines and the trained baseline.

Answers use Laya's record format (`laya_run.compact`), keyed `<variant>/<question>`, so one
scorer serves every system and paired comparisons use the same test messages.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crisis_triage import questions as Q
from crisis_triage.data import PROCESSED
from crisis_triage.laya_run import load
from crisis_triage.metrics import (
    auc,
    best_threshold,
    binary_f1,
    brier,
    ece,
    f1_per_class,
    macro_f1,
    paired_bootstrap,
    reliability,
)
from crisis_triage.runs import test_split
from crisis_triage.scoring import dev_sample, mean_auc_on, multilabel_scores, top

REPS = 1000


def ci(fn, n):
    return paired_bootstrap(fn, n, reps=REPS)


def score_multilabel(
    track: str, variant: str, test_path: Path, dev_path: Path, uids: set[str] | None = None
):
    """Metrics for yes/no-per-label answers; thresholds are the best-F1 ones on the dev answers.

    Returns the metrics and a function of row indices giving mean AUC (for paired comparisons).
    """
    labels = list(Q.HAITI_NEEDS if track == "haiti_sms" else Q.HUMSET_SECTORS)
    df = test_split(track)
    if uids is not None:  # score only these messages (the ones every compared system answered)
        df = df[df["uid"].isin(uids)].reset_index(drop=True)
    recs = load(Path(test_path))
    answers = [recs[u]["answers"] for u in df["uid"]]
    y = np.array([[c in set(ls) for c in labels] for ls in df["labels"]])
    s = multilabel_scores(answers, variant, labels)

    dev = dev_sample(track)
    dev_recs = load(Path(dev_path))
    dev_s = multilabel_scores([dev_recs[u]["answers"] for u in dev["uid"]], variant, labels)
    dev_y = np.array([[c in set(ls) for c in labels] for ls in dev["labels"]])
    thr = [best_threshold(dev_y[:, j], dev_s[:, j]) for j in range(len(labels))]
    pred = s >= np.array(thr)

    n = len(df)
    mean_auc = mean_auc_on(y, s)

    def f1_at_dev(i):
        return float(np.mean([binary_f1(y[i, j], pred[i, j], 0.5) for j in range(len(labels))]))

    out = {
        "n": n,
        "positives": {c: int(y[:, j].sum()) for j, c in enumerate(labels)},
        "mean_auc": ci(mean_auc, n),
        "macro_f1_dev_thresholds": ci(f1_at_dev, n),
        "per_label": {
            c: {
                "auc": auc(y[:, j], s[:, j]),
                "f1": binary_f1(y[:, j], pred[:, j], 0.5),
                "threshold": thr[j],
            }
            for j, c in enumerate(labels)
        },
        # Calibration of P(true) over every (message, label) pair.
        "ece": ece(s.ravel(), y.ravel()),
        "brier": brier(s.ravel(), y.ravel()),
        "reliability": reliability(s.ravel(), y.ravel()),
        "by_lang": {
            lang: {"n": int(m.sum()), "mean_auc": mean_auc(np.where(m)[0])}
            for lang, m in (
                (lg, (df["lang"] == lg).to_numpy()) for lg in sorted(df["lang"].unique())
            )
        },
        "routed_to": pd.Series([recs[u]["routed_to"] for u in df["uid"]]).value_counts().to_dict(),
    }
    if track in Q.URGENT_CLASSES:
        urgent = y[:, [labels.index(c) for c in Q.URGENT_CLASSES[track]]].any(axis=1)
        u = np.array([a[f"{Q.URGENCY_CHOSEN}/score"]["score"] for a in answers])
        out["urgency_auc"] = ci(lambda i: auc(urgent[i], u[i]), n)
        out["urgent_rate"] = float(urgent.mean())
    return out, mean_auc


def score_choice(track: str, variant: str, test_path: Path, uids: set[str] | None = None):
    """Metrics for one choice over the shared classes. Returns the metrics and a function of row
    indices giving macro-F1 (for paired comparisons)."""
    labels = list(Q.COARSE)
    names = {Q.readable(c): c for c in labels}
    df = test_split(track)
    if uids is not None:  # score only these messages (the ones every compared system answered)
        df = df[df["uid"].isin(uids)].reset_index(drop=True)
    recs = load(Path(test_path))
    answers = [recs[u]["answers"] for u in df["uid"]]
    probs = np.array(
        [[a[f"{variant}/label"]["probs"][Q.readable(c)] for c in labels] for a in answers]
    )
    gold = df["label"].map(Q.TO_COARSE).to_numpy()
    pred = np.array([names[top(a[f"{variant}/label"]["probs"])] for a in answers])
    conf = probs.max(axis=1)
    correct = pred == gold
    present = sorted(set(gold))
    n = len(df)

    # Reference point: always answer the most common dev class.
    dev = pd.read_parquet(PROCESSED / f"{track}.parquet").query("split == 'dev'")
    majority = dev["label"].map(Q.TO_COARSE).value_counts().idxmax()
    maj = np.full(n, majority)

    out = {
        "n": n,
        "classes": pd.Series(gold).value_counts().to_dict(),
        "macro_f1": ci(lambda i: macro_f1(gold[i], pred[i], present), n),
        "accuracy": ci(lambda i: float(correct[i].mean()), n),
        "per_class_f1": f1_per_class(gold, pred, present),
        "majority_class": {
            "class": majority,
            "macro_f1": macro_f1(gold, maj, present),
            "accuracy": float(np.mean(gold == maj)),
        },
        "ece": ece(conf, correct),
        "brier": brier(probs, np.array([[g == c for c in labels] for g in gold])),
        "reliability": reliability(conf, correct),
        "by_lang": {},
        "by_event": {},
        "routed_to": pd.Series([recs[u]["routed_to"] for u in df["uid"]]).value_counts().to_dict(),
    }
    for col, key in (("lang", "by_lang"), ("event", "by_event")):
        for value, part in df.groupby(col):
            idx = part.index.to_numpy()
            out[key][value] = {
                "n": len(idx),
                "macro_f1": macro_f1(gold[idx], pred[idx], sorted(set(gold[idx]))),
                "accuracy": float(correct[idx].mean()),
            }
    urgent = df["label"].isin(Q.URGENT_CLASSES[track]).to_numpy()
    u = np.array([a[f"{Q.URGENCY_CHOSEN}/score"]["score"] for a in answers])
    out["urgency_auc"] = ci(lambda i: auc(urgent[i], u[i]), n)
    out["urgent_rate"] = float(urgent.mean())
    return out, lambda i: macro_f1(gold[i], pred[i], present)
