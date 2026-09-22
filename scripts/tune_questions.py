"""T03: compare question wordings, question forms and checkpoints on a dev sample.

Runs Laya on gpu-box (`--run`), then scores the saved answers (`--score`, runs anywhere).
Answers: outputs/t03/<track>__<view>.jsonl. Scores: results/t03_dev.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crisis_triage import questions as Q
from crisis_triage.data import PROCESSED
from crisis_triage.laya_run import load, run
from crisis_triage.metrics import auc, binary_f1, macro_f1, paired_bootstrap

N_DEV = 1000
OUT = Path("outputs/t03")

# view name -> (text column, checkpoint; None = Laya's router decides)
VIEWS = {
    "haiti_sms": {
        "orig_router": ("text", None),
        "orig_multilingual": ("text", "multilingual"),
        "orig_english": ("text", "english"),
        "en_english": ("text_en", "english"),
        "en_typed": ("text_en", "typed-decisions"),
    },
    "humaid": {"router": ("text", None), "typed": ("text", "typed-decisions")},
    "crisisbench_ml": {"router": ("text", None), "multilingual": ("text", "multilingual")},
    "humset": {"router": ("text", None), "multilingual": ("text", "multilingual")},
}


def dev_sample(track: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / f"{track}.parquet")
    dev = df[df["split"] == "dev"]
    return dev.sample(min(N_DEV, len(dev)), random_state=0).reset_index(drop=True)


def class_labels(track: str) -> list[str]:
    df = pd.read_parquet(PROCESSED / f"{track}.parquet")
    return sorted(df["label"].unique())


def track_questions(track: str) -> dict:
    if track == "haiti_sms":
        return Q.flatten(Q.haiti_variants(), Q.URGENCY)
    if track == "humset":
        return Q.flatten(Q.humset_variants())
    return Q.flatten(Q.class_variants(class_labels(track)), Q.URGENCY)


def run_all(tracks=None):
    from laya import Router

    router = Router(preload=True, device="cuda")
    tok = router.load("english").tok
    budget = {}
    for track, views in VIEWS.items():
        if tracks and track not in tracks:
            continue
        qs = track_questions(track)
        budget[track] = {k: Q.head_tokens(tok, q) for k, q in qs.items()}
        rows = dev_sample(track).to_dict("records")
        for view, (col, model) in views.items():
            path = OUT / f"{track}__{view}.jsonl"
            print(track, view, len(rows), flush=True)
            run(router, rows, qs, col, model, path)
    path = OUT / "head_tokens.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(old | budget, indent=2))


# ---- scoring -------------------------------------------------------------------------------


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


def score_track(track: str) -> dict:
    df = dev_sample(track).set_index("uid")
    result = {"n": len(df), "views": {}}
    per_row = {}  # (view, variant) -> row metric inputs, for paired comparisons
    for view in VIEWS[track]:
        path = OUT / f"{track}__{view}.jsonl"
        if not path.exists():
            continue
        recs = load(path)
        uids = [u for u in df.index if u in recs]
        answers = [recs[u]["answers"] for u in uids]
        routed = pd.Series([recs[u]["routed_to"] for u in uids]).value_counts().to_dict()
        v = {"n": len(uids), "routed_to": routed, "variants": {}}
        sub = df.loc[uids]
        if track in ("haiti_sms", "humset"):
            labels = list(Q.HAITI_NEEDS if track == "haiti_sms" else Q.HUMSET_SECTORS)
            y = np.array([[c in set(ls) for c in labels] for ls in sub["labels"]])
            variants = Q.haiti_variants() if track == "haiti_sms" else Q.humset_variants()
            for var in variants:
                s = multilabel_scores(answers, var, labels)
                v["variants"][var] = score_multilabel(y, s, labels)
                per_row[(view, var)] = (y, s)
        else:
            fine = class_labels(track)
            for var in Q.class_variants(fine):
                coarse = var.startswith("coarse")
                labels = list(Q.COARSE) if coarse else fine
                names = {Q.readable(c): c for c in labels}
                gold = sub["label"].map(Q.TO_COARSE if coarse else (lambda c: c)).to_numpy()
                pred = np.array([names[top(a[f"{var}/label"]["probs"])] for a in answers])
                present = sorted(set(gold))
                v["variants"][var] = {
                    "macro_f1": macro_f1(gold, pred, present),
                    "accuracy": float(np.mean(gold == pred)),
                }
                per_row[(view, var)] = (gold, pred, present)
        if track in Q.URGENT_CLASSES:
            urgent_set = Q.URGENT_CLASSES[track]
            if track == "haiti_sms":
                urgent = np.array([bool(set(ls) & urgent_set) for ls in sub["labels"]])
            else:
                urgent = sub["label"].isin(urgent_set).to_numpy()
            v["urgency_auc"] = {
                k: auc(urgent, [a[f"{k}/score"]["score"] for a in answers]) for k in Q.URGENCY
            }
            v["urgent_rate"] = float(urgent.mean())
        result["views"][view] = v
    result["best"], result["vs_runner_up"] = compare(track, per_row)
    return result


def compare(track, per_row):
    """Best (view, variant) on dev and the paired bootstrap difference to the runner-up."""
    if not per_row:
        return None, None
    if track in ("haiti_sms", "humset"):
        fns = {k: mean_auc_on(y, s) for k, (y, s) in per_row.items()}
    else:
        fns = {
            k: (lambda g, p, pr: lambda i: macro_f1(g[i], p[i], pr))(*t) for k, t in per_row.items()
        }
    n = len(next(iter(per_row.values()))[0])
    if track in ("humaid", "crisisbench_ml"):
        # coarse and fine are different questions: pick among the coarse ones (the one used)
        fns = {k: f for k, f in fns.items() if k[1].startswith("coarse")}
    ranked = sorted(fns, key=lambda k: fns[k](np.arange(n)), reverse=True)
    if len(ranked) < 2:
        return "/".join(ranked[0]), None
    a, b = ranked[0], ranked[1]
    if len(per_row[a][0]) != len(per_row[b][0]):
        return "/".join(a), None
    diff = paired_bootstrap(lambda i: fns[a](i) - fns[b](i), n, reps=500)
    return "/".join(a), {"runner_up": "/".join(b), "diff": diff}


# The comparisons behind each default, as (track, (view, variant) A, (view, variant) B, metric).
DECISIONS = {
    "haiti: forced multilingual vs router, original text": (
        "haiti_sms",
        ("orig_multilingual", "noul_described"),
        ("orig_router", "noul_described"),
    ),
    "haiti: English translation (typed) vs original (multilingual)": (
        "haiti_sms",
        ("en_typed", "noul_described"),
        ("orig_multilingual", "noul_described"),
    ),
    "haiti: typed vs english checkpoint, translation": (
        "haiti_sms",
        ("en_typed", "noul_described"),
        ("en_english", "noul_described"),
    ),
    "haiti: one noul per need vs one choice": (
        "haiti_sms",
        ("en_typed", "noul_described"),
        ("en_typed", "choice_described"),
    ),
    "haiti: described vs short noul": (
        "haiti_sms",
        ("en_typed", "noul_described"),
        ("en_typed", "noul_short"),
    ),
    "humaid: typed vs router": (
        "humaid",
        ("typed", "coarse_described"),
        ("router", "coarse_described"),
    ),
    "humaid: coarse described vs coarse names": (
        "humaid",
        ("typed", "coarse_described"),
        ("typed", "coarse_names"),
    ),
    "crisisbench_ml: router vs forced multilingual": (
        "crisisbench_ml",
        ("router", "coarse_described"),
        ("multilingual", "coarse_described"),
    ),
    "humset: router vs forced multilingual": (
        "humset",
        ("router", "noul_described"),
        ("multilingual", "noul_described"),
    ),
    "humset: described vs short, router": (
        "humset",
        ("router", "noul_described"),
        ("router", "noul_short"),
    ),
}


def rows_for(track, view, variant):
    """Aligned (gold, scores-or-preds) for one view/variant on the dev sample."""
    df = dev_sample(track).set_index("uid")
    recs = load(OUT / f"{track}__{view}.jsonl")
    answers = [recs[u]["answers"] for u in df.index]
    if track in ("haiti_sms", "humset"):
        labels = list(Q.HAITI_NEEDS if track == "haiti_sms" else Q.HUMSET_SECTORS)
        y = np.array([[c in set(ls) for c in labels] for ls in df["labels"]])
        return mean_auc_on(y, multilabel_scores(answers, variant, labels)), len(df), "mean_auc"
    names = {Q.readable(c): c for c in Q.COARSE}
    gold = df["label"].map(Q.TO_COARSE).to_numpy()
    pred = np.array([names[top(a[f"{variant}/label"]["probs"])] for a in answers])
    present = sorted(set(gold))
    return (lambda i: macro_f1(gold[i], pred[i], present)), len(df), "macro_f1"


def urgency_decision():
    """Concrete vs plain urgency wording, per track, paired on the default view."""
    out = {}
    for track, view in (("haiti_sms", "en_typed"), ("humaid", "typed")):
        df = dev_sample(track).set_index("uid")
        recs = load(OUT / f"{track}__{view}.jsonl")
        urgent_set = Q.URGENT_CLASSES[track]
        if track == "haiti_sms":
            y = np.array([bool(set(ls) & urgent_set) for ls in df["labels"]])
        else:
            y = df["label"].isin(urgent_set).to_numpy()
        s = {
            k: np.array([recs[u]["answers"][f"{k}/score"]["score"] for u in df.index])
            for k in Q.URGENCY
        }
        out[track] = paired_bootstrap(
            lambda i, y=y, s=s: (
                auc(y[i], s["urgency_concrete"][i]) - auc(y[i], s["urgency_plain"][i])
            ),
            len(y),
            reps=500,
        )
    return out


def decisions():
    out = {}
    for name, (track, a, b) in DECISIONS.items():
        fa, n, metric = rows_for(track, *a)
        fb, _, _ = rows_for(track, *b)
        out[name] = {
            "metric": metric,
            "a": round(fa(np.arange(n)), 4),
            "b": round(fb(np.arange(n)), 4),
            "a_minus_b": paired_bootstrap(lambda i, fa=fa, fb=fb: fa(i) - fb(i), n, reps=500),
        }
    out["urgency: concrete vs plain wording (AUC)"] = urgency_decision()
    return out


def score_all():
    out = {t: score_track(t) for t in VIEWS}
    out["decisions"] = decisions()
    budget = OUT / "head_tokens.json"
    if budget.exists():
        out["head_tokens"] = json.loads(budget.read_text())
    Path("results").mkdir(exist_ok=True)
    Path("results/t03_dev.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--tracks", nargs="*", help="limit --run to these tracks")
    args = parser.parse_args()
    if args.run:
        run_all(args.tracks)
    if args.score:
        print(json.dumps(score_all(), indent=1, default=float)[:6000])
