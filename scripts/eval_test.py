"""T04: Laya zero-shot on the test splits, with every setting chosen on dev (T03, T03b).

`--run` (gpu-box): translate the Haiti test SMS with the chosen NLLB setting, then answer the
chosen questions for every test message. `--score`: metrics with 95% bootstrap intervals,
calibration, and thresholds taken from the dev answers. Results: results/t04_test.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crisis_triage import questions as Q
from crisis_triage.data import PROCESSED
from crisis_triage.laya_run import load, run
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
from crisis_triage.runs import RUNS, SETTINGS, run_questions, test_split
from crisis_triage.scoring import dev_sample, mean_auc_on, multilabel_scores, top
from crisis_triage.translate import CHOSEN_MT, NLLB_CODES, Translator

OUT = Path("outputs/t04")
REPS = 1000


def translate_haiti() -> dict[str, str]:
    path = OUT / "translations_haiti_test.jsonl"  # not haiti_mt.jsonl: that holds the answers
    if not path.exists():
        df = test_split("haiti_sms")
        tr = Translator(CHOSEN_MT["model"], num_beams=CHOSEN_MT["num_beams"])
        with path.open("w") as f:
            for lang, part in df.groupby("lang"):
                rows = part.to_dict("records")
                for i in range(0, len(rows), 32):
                    chunk = rows[i : i + 32]
                    outs = tr([r["text"] for r in chunk], NLLB_CODES[lang], CHOSEN_MT["lowercase"])
                    for r, t in zip(chunk, outs, strict=True):
                        f.write(json.dumps({"uid": r["uid"], "text_mt": t}, ensure_ascii=False))
                        f.write("\n")
        del tr
    return {r["uid"]: r["text_mt"] for r in map(json.loads, path.open())}


def run_all():
    import torch
    from laya import Router

    OUT.mkdir(parents=True, exist_ok=True)
    mt = translate_haiti()
    torch.cuda.empty_cache()
    router = Router(preload=True, device="cuda")  # router runs switch checkpoints per message
    for name, (track, setting, _) in RUNS.items():
        cfg = SETTINGS[setting]
        df = test_split(track)
        if cfg["text"] == "text_mt":
            df["text_mt"] = df["uid"].map(mt)
        print(name, len(df), flush=True)
        qs = run_questions(track, cfg["variant"])
        rows = df[["uid", cfg["text"]]].to_dict("records")
        run(router, rows, qs, cfg["text"], cfg["model"], OUT / f"{name}.jsonl")


# ---- scoring -------------------------------------------------------------------------------


def ci(fn, n):
    return paired_bootstrap(fn, n, reps=REPS)


def score_multilabel_run(name, track, setting, dev_path) -> tuple[dict, callable]:
    variant = SETTINGS[setting]["variant"]
    labels = list(Q.HAITI_NEEDS if track == "haiti_sms" else Q.HUMSET_SECTORS)
    df = test_split(track)
    recs = load(OUT / f"{name}.jsonl")
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


def score_choice_run(name, track, setting) -> dict:
    variant = SETTINGS[setting]["variant"]
    labels = list(Q.COARSE)
    names = {Q.readable(c): c for c in labels}
    df = test_split(track)
    recs = load(OUT / f"{name}.jsonl")
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
    return out


def score_all() -> dict:
    results, auc_fns = {}, {}
    for name, (track, setting, dev_path) in RUNS.items():
        if not (OUT / f"{name}.jsonl").exists():
            continue
        if dev_path:
            results[name], auc_fns[name] = score_multilabel_run(name, track, setting, dev_path)
        else:
            results[name] = score_choice_run(name, track, setting)
        print(name, "scored", flush=True)
    n = results["haiti_mt"]["n"]
    results["haiti_paired_mean_auc"] = {
        "mt - original": ci(lambda i: auc_fns["haiti_mt"](i) - auc_fns["haiti_original"](i), n),
        "human - mt": ci(lambda i: auc_fns["haiti_human"](i) - auc_fns["haiti_mt"](i), n),
    }
    Path("results/t04_test.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.run:
        run_all()
    if args.score:
        score_all()
