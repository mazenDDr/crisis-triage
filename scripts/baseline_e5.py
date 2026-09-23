"""T05 trained baseline: e5 embeddings + logistic regression, trained on each dev split.

Writes outputs/t05/e5lr/<run>__test.jsonl (every test message) and <run>__dev.jsonl (the fixed
dev sample, out-of-fold, used only for thresholds), plus one-message-at-a-time timing.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from crisis_triage import questions as Q
from crisis_triage import trained as T
from crisis_triage.data import PROCESSED
from crisis_triage.latency import machine_label, summarize
from crisis_triage.runs import RUNS, SETTINGS
from crisis_triage.scoring import dev_sample

OUT = Path("outputs/t05/e5lr")
E5_RUNS = ("haiti_original", "humaid", "crisisbench_ml", "humset")


def train_rows(track: str) -> pd.DataFrame:
    """The dev split; for big ones a seeded sample that always keeps the fixed dev sample."""
    df = pd.read_parquet(PROCESSED / f"{track}.parquet")
    dev = df[df["split"] == "dev"]
    if len(dev) <= T.MAX_TRAIN:
        return dev.reset_index(drop=True)
    keep = set(dev_sample(track)["uid"])
    rest = dev[~dev["uid"].isin(keep)].sample(T.MAX_TRAIN - len(keep), random_state=0)
    return pd.concat([dev[dev["uid"].isin(keep)], rest]).reset_index(drop=True)


def write(path: Path, uids, answers):
    with path.open("w") as f:
        for u, a in zip(uids, answers, strict=True):
            f.write(json.dumps({"uid": u, "routed_to": "e5-lr", "answers": a}) + "\n")


def main():
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    emb = T.Embedder()
    timing = {}
    for name in E5_RUNS:
        track, setting, _ = RUNS[name]
        variant = SETTINGS[setting]["variant"]
        tr = train_rows(track)
        df = pd.read_parquet(PROCESSED / f"{track}.parquet")
        te = df[df["split"] == "test"].reset_index(drop=True)
        x, x_te = emb(tr["text"].tolist()), emb(te["text"].tolist())
        if track in ("haiti_sms", "humset"):
            labels = list(Q.HAITI_NEEDS if track == "haiti_sms" else Q.HUMSET_SECTORS)
            y = np.array([[c in set(ls) for c in labels] for ls in tr["labels"]])
            p_te, p_oof = T.fit_multilabel(x, y, x_te)
            urgent = Q.URGENT_CLASSES.get(track)
            ans_te = T.multilabel_answers(p_te, labels, variant, urgent)
            ans_oof = T.multilabel_answers(p_oof, labels, variant, urgent)
            predict = T.classifier().fit(x, y[:, 0])  # one head, for timing
        else:
            y = tr["label"].map(Q.TO_COARSE).to_numpy()
            p_te, p_oof = T.fit_choice(x, y, x_te)
            ans_te, ans_oof = T.choice_answers(p_te, variant), T.choice_answers(p_oof, variant)
            predict = T.classifier().fit(x, y)
        write(OUT / f"{name}__test.jsonl", te["uid"], ans_te)
        keep = set(dev_sample(track)["uid"])
        idx = [i for i, u in enumerate(tr["uid"]) if u in keep]
        write(OUT / f"{name}__dev.jsonl", tr["uid"].iloc[idx], [ans_oof[i] for i in idx])

        # One message at a time: embed + classify (multi-label: every head).
        heads = len(labels) if track in ("haiti_sms", "humset") else 1
        texts = te["text"].sample(110, random_state=0).tolist()
        times = []
        for k, t in enumerate(texts):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            v = emb([t], batch_size=1)
            for _ in range(heads):
                predict.predict_proba(v)
            torch.cuda.synchronize()
            if k >= 10:
                times.append((time.perf_counter() - t0) * 1e3)
        timing[name] = {"train_rows": len(tr), "heads": heads, **summarize(times)}
        print(name, timing[name], flush=True)
    (OUT / "timing.json").write_text(
        json.dumps({"machine": machine_label(), "warmup": 10, "runs": timing}, indent=2)
    )


if __name__ == "__main__":
    main()
