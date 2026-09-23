"""The cheap trained baseline: multilingual-e5 sentence embeddings + logistic regression.

Trained on each track's dev split only (the past disasters), so it sees exactly the data Laya's
settings were tuned on, plus the labels. Its dev answers come from 5-fold cross-validation, so
thresholds chosen on them are not fitted to the same messages. Answers are written in Laya's
format so `evaluate` scores every system the same way.
"""

from __future__ import annotations

import numpy as np

from crisis_triage import questions as Q

E5 = "intfloat/multilingual-e5-base"
MAX_TRAIN = 40_000  # HumSet dev has 131k excerpts; a seeded sample keeps training quick


class Embedder:
    def __init__(self, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(E5, device=device)
        if device == "cuda":
            self.model.half()

    def __call__(self, texts, batch_size: int = 128) -> np.ndarray:
        return self.model.encode(
            [f"query: {t}" for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


# Settings compared by cross-validation on dev: regularisation strength and class weighting.
# (The first T05 run used the scikit-learn default C=1 without weighting, which is heavily
# regularised for normalised embeddings and badly under-fits small or imbalanced sets.)
GRID = [
    {"C": c, "class_weight": w} for c in (0.3, 1.0, 3.0, 10.0, 30.0) for w in (None, "balanced")
]
SELECT_MAX_ROWS = 10_000  # multi-label selection runs on a seeded sample this large


def classifier(C: float = 1.0, class_weight: str | None = None):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=3000, C=C, class_weight=class_weight)


def select_choice(x: np.ndarray, labels: np.ndarray, folds: int = 5) -> tuple[dict, list[dict]]:
    """The GRID setting with the best out-of-fold macro-F1 on (x, labels), and every score."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from crisis_triage.metrics import macro_f1

    cv = StratifiedKFold(folds, shuffle=True, random_state=0)
    table = []
    for params in GRID:
        oof = cross_val_predict(classifier(**params), x, labels, cv=cv)
        table.append({**params, "cv_macro_f1": macro_f1(labels, oof, sorted(set(labels)))})
    best = max(table, key=lambda r: r["cv_macro_f1"])
    return {k: best[k] for k in ("C", "class_weight")}, table


def select_multilabel(x: np.ndarray, y: np.ndarray, folds: int = 3) -> tuple[dict, list[dict]]:
    """The GRID setting with the best out-of-fold mean AUC over labels, and every score."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from crisis_triage.metrics import auc

    if len(x) > SELECT_MAX_ROWS:
        keep = np.random.default_rng(0).choice(len(x), SELECT_MAX_ROWS, replace=False)
        x, y = x[keep], y[keep]
    table = []
    for params in GRID:
        aucs = []
        for j in range(y.shape[1]):
            cv = StratifiedKFold(folds, shuffle=True, random_state=0)
            p = cross_val_predict(classifier(**params), x, y[:, j], cv=cv, method="predict_proba")
            aucs.append(auc(y[:, j], p[:, 1]))
        table.append({**params, "cv_mean_auc": float(np.nanmean(aucs))})
    best = max(table, key=lambda r: r["cv_mean_auc"])
    return {k: best[k] for k in ("C", "class_weight")}, table


def fit_multilabel(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, params: dict, folds: int = 5):
    """P(label) for new rows (fit on all x) and out-of-fold P(label) for x itself."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    new, oof = np.zeros((len(x_new), y.shape[1])), np.zeros(y.shape)
    for j in range(y.shape[1]):
        cv = StratifiedKFold(folds, shuffle=True, random_state=0)
        oof[:, j] = cross_val_predict(
            classifier(**params), x, y[:, j], cv=cv, method="predict_proba"
        )[:, 1]
        new[:, j] = classifier(**params).fit(x, y[:, j]).predict_proba(x_new)[:, 1]
    return new, oof


def fit_choice(x: np.ndarray, labels: np.ndarray, x_new: np.ndarray, params: dict, folds: int = 5):
    """(n, len(COARSE)) probabilities for new rows and out-of-fold for x. Classes missing from
    the training labels get probability 0: the model cannot predict what it never saw."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    classes = list(Q.COARSE)
    clf = classifier(**params).fit(x, labels)
    cols = [classes.index(c) for c in clf.classes_]
    new = np.zeros((len(x_new), len(classes)))
    new[:, cols] = clf.predict_proba(x_new)
    cv = StratifiedKFold(folds, shuffle=True, random_state=0)
    oof = np.zeros((len(x), len(classes)))
    oof[:, cols] = cross_val_predict(classifier(**params), x, labels, cv=cv, method="predict_proba")
    return new, oof


def multilabel_answers(p: np.ndarray, labels: list[str], variant: str, urgent: set[str] | None):
    """Records in Laya's format. Urgency (0-2) = 2 x the highest P among the urgent needs."""
    out = []
    for row in p:
        a = {f"{variant}/{c}": {"p": float(v)} for c, v in zip(labels, row, strict=True)}
        if urgent:
            u = max(row[labels.index(c)] for c in urgent)
            a[f"{Q.URGENCY_CHOSEN}/score"] = {"score": 2 * float(u)}
        out.append(a)
    return out


# Coarse classes whose fine labels make up most of the urgency stand-in.
URGENT_COARSE = ("requests_or_needs", "people_affected")


def choice_answers(p: np.ndarray, variant: str):
    """Records in Laya's format. Urgency (0-2) = 2 x P(requests or people affected)."""
    classes = list(Q.COARSE)
    out = []
    for row in p:
        probs = {Q.readable(c): float(v) for c, v in zip(classes, row, strict=True)}
        u = sum(row[classes.index(c)] for c in URGENT_COARSE)
        out.append(
            {
                f"{variant}/label": {"probs": probs},
                f"{Q.URGENCY_CHOSEN}/score": {"score": 2 * float(u)},
            }
        )
    return out
