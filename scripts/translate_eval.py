"""T03b: NLLB translation of the Haiti SMS, then Laya, on the T03 dev sample.

`--run` (gpu-box): translate with each model, time one message at a time, run Laya with the
chosen questions on the translations. `--score`: compare with the T03 runs on the same
messages (original text, human translation). Results: results/t03b_dev.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from crisis_triage import questions as Q
from crisis_triage.latency import machine_label, summarize
from crisis_triage.laya_run import load, run
from crisis_triage.metrics import auc, paired_bootstrap
from crisis_triage.scoring import dev_sample, mean_auc_on, multilabel_scores, score_multilabel
from crisis_triage.translate import NLLB_CODES, Translator

MODELS = {
    "nllb600m": "facebook/nllb-200-distilled-600M",
    "nllb1.3b": "facebook/nllb-200-distilled-1.3B",
}
# Translation settings compared: model x lower-casing the SMS first x beam search width.
CONFIGS = {
    f"{m}{suffix}": (m, low, 4) for m in MODELS for suffix, low in (("", False), ("_lower", True))
}
CONFIGS["nllb600m_lower_greedy"] = ("nllb600m", True, 1)
OUT = Path("outputs/t03b")
T03 = Path("outputs/t03")
CHOSEN = Q.CHOSEN["haiti_sms"]
TIMED = 100  # messages timed one at a time


def questions() -> dict:
    variant = CHOSEN["variant"]
    return Q.flatten(
        {variant: Q.haiti_variants()[variant]}, {Q.URGENCY_CHOSEN: Q.URGENCY[Q.URGENCY_CHOSEN]}
    )


def translate(config: str, df) -> dict:
    """Translate the dev sample with one setting (skipped if saved), then time it."""
    import torch

    model, low, beams = CONFIGS[config]
    tr = Translator(MODELS[model], num_beams=beams)
    path = OUT / f"mt_{config}.jsonl"
    if not path.exists():
        with path.open("w") as f:
            for lang, part in df.groupby("lang"):
                rows = part.to_dict("records")
                for i in range(0, len(rows), 32):
                    chunk = rows[i : i + 32]
                    outs = tr([r["text"] for r in chunk], NLLB_CODES[lang], lowercase=low)
                    for r, t in zip(chunk, outs, strict=True):
                        f.write(json.dumps({"uid": r["uid"], "text_mt": t}, ensure_ascii=False))
                        f.write("\n")
    # One message at a time, like a live SMS line; 10 warm-up messages first.
    rows = df.head(TIMED + 10).to_dict("records")
    times = []
    for k, r in enumerate(rows):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tr([r["text"]], NLLB_CODES[r["lang"]], lowercase=low)
        torch.cuda.synchronize()
        if k >= 10:
            times.append((time.perf_counter() - t0) * 1e3)
    del tr
    torch.cuda.empty_cache()
    return {"machine": machine_label(), "warmup": 10, "runs": TIMED, **summarize(times)}


def run_all():
    from laya import Router

    OUT.mkdir(parents=True, exist_ok=True)
    df = dev_sample("haiti_sms")
    timing = {config: translate(config, df) for config in CONFIGS}
    (OUT / "mt_timing.json").write_text(json.dumps(timing, indent=2))
    router = Router(preload=False, device="cuda")
    qs = questions()
    for config in CONFIGS:
        mt = {r["uid"]: r["text_mt"] for r in map(json.loads, (OUT / f"mt_{config}.jsonl").open())}
        rows = [{"uid": u, "text_mt": mt[u]} for u in df["uid"]]
        run(router, rows, qs, "text_mt", CHOSEN["model"], OUT / f"laya_{config}.jsonl")


def score_all() -> dict:
    df = dev_sample("haiti_sms").set_index("uid")
    labels = list(Q.HAITI_NEEDS)
    y = np.array([[c in set(ls) for c in labels] for ls in df["labels"]])
    urgent = np.array([bool(set(ls) & Q.URGENT_CLASSES["haiti_sms"]) for ls in df["labels"]])
    sources = {
        "original": T03 / "haiti_sms__orig_multilingual.jsonl",
        **{config: OUT / f"laya_{config}.jsonl" for config in CONFIGS},
        "human_translation": T03 / "haiti_sms__en_typed.jsonl",
    }
    variant = CHOSEN["variant"]
    views, fns = {}, {}
    for name, path in sources.items():
        recs = load(path)
        answers = [recs[u]["answers"] for u in df.index]
        s = multilabel_scores(answers, variant, labels)
        u = [a[f"{Q.URGENCY_CHOSEN}/score"]["score"] for a in answers]
        views[name] = {**score_multilabel(y, s, labels), "urgency_auc": auc(urgent, u)}
        fns[name] = mean_auc_on(y, s)
    n = len(df)
    best_mt = max(CONFIGS, key=lambda m: views[m]["mean_auc"])
    pairs = {
        f"{best_mt} - original": (best_mt, "original"),
        f"human_translation - {best_mt}": ("human_translation", best_mt),
        "nllb1.3b_lower - nllb600m_lower": ("nllb1.3b_lower", "nllb600m_lower"),
        "nllb600m_lower - nllb600m": ("nllb600m_lower", "nllb600m"),
        "nllb1.3b_lower - nllb1.3b": ("nllb1.3b_lower", "nllb1.3b"),
        "nllb600m_lower_greedy - nllb600m_lower": ("nllb600m_lower_greedy", "nllb600m_lower"),
    }
    diffs = {
        k: paired_bootstrap(lambda i, a=a, b=b: fns[a](i) - fns[b](i), n, reps=500)
        for k, (a, b) in pairs.items()
    }
    timing = json.loads((OUT / "mt_timing.json").read_text())
    out = {
        "n": n,
        "variant": variant,
        "best_mt": best_mt,
        "views": views,
        "paired_mean_auc": diffs,
        "mt_ms": timing,
    }
    Path("results/t03b_dev.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.run:
        run_all()
    if args.score:
        print(json.dumps(score_all(), indent=1, default=float))
