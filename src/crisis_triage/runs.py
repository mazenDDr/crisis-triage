"""The T04 test runs: which track, setting and question set each one uses.

Shared by the evaluation (scripts/eval_test.py) and the speed benchmark (scripts/time_laya.py)
so both measure exactly the same configuration.
"""

from __future__ import annotations

import pandas as pd

from crisis_triage import questions as Q
from crisis_triage.data import PROCESSED

# Each run: track, the chosen setting (questions.CHOSEN key), and the dev answers used to pick
# thresholds (multi-label runs only).
RUNS = {
    "haiti_original": (
        "haiti_sms",
        "haiti_sms_original",
        "outputs/t03/haiti_sms__orig_multilingual.jsonl",
    ),
    "haiti_mt": ("haiti_sms", "haiti_sms_mt", "outputs/t03b/laya_nllb600m_lower.jsonl"),
    "haiti_human": ("haiti_sms", "haiti_sms", "outputs/t03/haiti_sms__en_typed.jsonl"),
    "humaid": ("humaid", "humaid", None),
    "crisisbench_ml": ("crisisbench_ml", "crisisbench_ml", None),
    "humset": ("humset", "humset", "outputs/t03/humset__router.jsonl"),
}
SETTINGS = Q.CHOSEN | {"haiti_sms_mt": {**Q.CHOSEN["haiti_sms"], "text": "text_mt"}}


def test_split(track: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / f"{track}.parquet")
    return df[df["split"] == "test"].reset_index(drop=True)


def run_questions(track: str, variant: str) -> dict:
    if track == "haiti_sms":
        variants = Q.haiti_variants()
    elif track == "humset":
        variants = Q.humset_variants()
    elif variant.startswith("coarse"):
        # the shared class set: no need to read the data (the Mac timing run has none)
        wording = variant.removeprefix("coarse_")
        variants = {variant: {"label": Q.choice_question(list(Q.COARSE), wording, Q.COARSE)}}
    else:
        labels = sorted(pd.read_parquet(PROCESSED / f"{track}.parquet")["label"].unique())
        variants = Q.class_variants(labels)
    urgency = {} if track == "humset" else {Q.URGENCY_CHOSEN: Q.URGENCY[Q.URGENCY_CHOSEN]}
    return Q.flatten({variant: variants[variant]}, urgency)
