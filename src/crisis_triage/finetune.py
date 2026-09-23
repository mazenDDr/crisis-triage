"""Training examples for fine-tuning Laya on the dev splits (T07).

Each example is one (message, question) pair with a target distribution over the question's
options, the format Laya's own training recipe uses. Only dev messages outside the fixed dev
sample are used: the dev sample is held out for the temperature fit and the thresholds, and
test is never touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crisis_triage import questions as Q
from crisis_triage.data import PROCESSED
from crisis_triage.runs import RUNS, SETTINGS, run_questions
from crisis_triage.scoring import dev_sample

# Messages per track used for training (Haiti and HumSet ask 4 and 10 questions per message).
CAPS = {"haiti_sms": 4000, "humaid": 8000, "crisisbench_ml": None, "humset": 1500}
TRACK_RUN = {
    "haiti_sms": "haiti_human",
    "humaid": "humaid",
    "crisisbench_ml": "crisisbench_ml",
    "humset": "humset",
}


def target(qdef: dict, answer) -> list[float]:
    """One-hot target in Laya's option order: noul is [false, true]; choice follows criteria."""
    if qdef["type"] == "noul":
        return [0.0, 1.0] if answer else [1.0, 0.0]
    crit = qdef["criteria"]
    keys = list(crit) if isinstance(crit, dict) else list(crit)
    return [1.0 if k == answer else 0.0 for k in keys]


def examples(track: str, df: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    """(state, question, target) examples for one track. Haiti messages are shown either in the
    original language or in English (a coin flip per message), so the model learns both."""
    name = TRACK_RUN[track]
    _, setting, _ = RUNS[name]
    variant = SETTINGS[setting]["variant"]
    qs = {k: q for k, q in run_questions(track, variant).items() if not k.startswith("urgency")}
    out = []
    for r in df.to_dict("records"):
        text = r["text"]
        if track == "haiti_sms" and rng.random() < 0.5:
            text = r["text_en"]
        for key, qdef in qs.items():
            question = key.split("/", 1)[1]
            if track in ("haiti_sms", "humset"):
                answer = question in set(r["labels"])
            else:
                answer = Q.readable(Q.TO_COARSE[r["label"]])
            out.append({"state": {"message": text}, "q": qdef, "target": target(qdef, answer)})
    return out


def training_frames(seed: int = 0) -> dict[str, pd.DataFrame]:
    """Per track: dev messages outside the fixed dev sample, capped by CAPS (seeded)."""
    frames = {}
    for track, cap in CAPS.items():
        df = pd.read_parquet(PROCESSED / f"{track}.parquet")
        dev = df[(df["split"] == "dev") & ~df["uid"].isin(set(dev_sample(track)["uid"]))]
        if cap and len(dev) > cap:
            dev = dev.sample(cap, random_state=seed)
        frames[track] = dev.reset_index(drop=True)
    return frames
