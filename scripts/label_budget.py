"""T05c: how many labelled messages does a small trained classifier need to beat Laya out of
the box (and the fine-tuned Laya)?

e5 + logistic regression is trained on random draws of N dev messages (5 seeds per N) and
scored on every test message of the track. Each draw is compared with the Laya answers on
the same test messages (paired bootstrap on macro-F1). Two classifier settings are run: the
scikit-learn default (C=1, no weighting) and the setting chosen by dev cross-validation in
baseline_e5.py. The tuned setting was chosen with all dev labels, which slightly favours the
small budgets. Results: results/t05c_label_budget.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crisis_triage import questions as Q
from crisis_triage import trained as T
from crisis_triage.data import PROCESSED
from crisis_triage.laya_run import load
from crisis_triage.metrics import macro_f1, paired_bootstrap
from crisis_triage.scoring import top

BUDGETS = (25, 50, 100, 200, 400, 800, 1600, 3200, 6400)
SEEDS = 5
TRACKS = {"humaid": "coarse_described", "crisisbench_ml": "coarse_described"}
LAYA = {"laya": "outputs/t04/{run}.jsonl", "laya-ft": "outputs/t07/{run}__test.jsonl"}


def laya_preds(path: str, uids, variant: str) -> np.ndarray:
    recs = load(Path(path))
    names = {Q.readable(c): c for c in Q.COARSE}
    return np.array([names[top(recs[u]["answers"][f"{variant}/label"]["probs"])] for u in uids])


def main():
    emb = T.Embedder()
    tuned = json.loads(Path("outputs/t05/e5lr/timing.json").read_text())["runs"]
    out = {"budgets": list(BUDGETS), "seeds": SEEDS, "tracks": {}}
    for run, variant in TRACKS.items():
        df = pd.read_parquet(PROCESSED / f"{run}.parquet")
        dev = df[df["split"] == "dev"].reset_index(drop=True)
        te = df[df["split"] == "test"].reset_index(drop=True)
        x_dev, x_te = emb(dev["text"].tolist()), emb(te["text"].tolist())
        y_dev = dev["label"].map(Q.TO_COARSE).to_numpy()
        gold = te["label"].map(Q.TO_COARSE).to_numpy()
        present = sorted(set(gold))
        refs = {k: laya_preds(p.format(run=run), te["uid"], variant) for k, p in LAYA.items()}
        track = {
            "n_dev": len(dev),
            "n_test": len(te),
            "laya_macro_f1": {k: macro_f1(gold, p, present) for k, p in refs.items()},
            "settings": {},
        }
        settings = {"default": {}, "tuned": tuned[run]["params"]}
        for name, params in settings.items():
            budgets = {}
            for n in [b for b in BUDGETS if b < len(dev)] + [len(dev)]:
                rows = []
                for seed in range(SEEDS if n < len(dev) else 1):
                    idx = np.random.default_rng(seed).choice(len(dev), n, replace=False)
                    y = y_dev[idx]
                    clf = T.classifier(**params).fit(x_dev[idx], y) if len(set(y)) > 1 else None
                    pred = clf.predict(x_te) if clf else np.full(len(te), y[0])
                    f1 = macro_f1(gold, pred, present)
                    diffs = {
                        k: paired_bootstrap(
                            lambda i, p=pred, r=r, g=gold, pr=present: (
                                macro_f1(g[i], p[i], pr) - macro_f1(g[i], r[i], pr)
                            ),
                            len(te),
                            reps=200,
                            seed=seed,
                        )
                        for k, r in refs.items()
                    }
                    rows.append(
                        {"seed": seed, "classes_seen": len(set(y)), "macro_f1": f1, "minus": diffs}
                    )
                    print(run, name, n, seed, round(f1, 3), flush=True)
                f1s = [r["macro_f1"] for r in rows]
                budgets[str(n)] = {
                    "mean_macro_f1": float(np.mean(f1s)),
                    "min": float(np.min(f1s)),
                    "max": float(np.max(f1s)),
                    # a draw "beats" a Laya when its paired interval is entirely above 0
                    "beats": {k: sum(r["minus"][k]["lo"] > 0 for r in rows) for k in refs},
                    "loses": {k: sum(r["minus"][k]["hi"] < 0 for r in rows) for k in refs},
                    "draws": rows,
                }
            track["settings"][name] = {"params": params, "budgets": budgets}
        out["tracks"][run] = track
    Path("results/t05c_label_budget.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
