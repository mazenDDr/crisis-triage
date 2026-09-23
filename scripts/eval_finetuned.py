"""T07: run the fine-tuned Laya (data/laya_ft) on every T04 run, dev sample and test.

Same questions, texts and runner as zero-shot Laya; one checkpoint for every run. Writes
outputs/t07/<run>__<split>.jsonl and outputs/t07/timing.json (one message at a time).
"""

import json
import time
from pathlib import Path

import torch

from crisis_triage.latency import machine_label, summarize
from crisis_triage.laya_run import run
from crisis_triage.runs import RUNS, SETTINGS, run_questions, test_split
from crisis_triage.scoring import dev_sample

OUT = Path("outputs/t07")
MT = {
    "dev": Path("outputs/t03b/mt_nllb600m_lower.jsonl"),
    "test": Path("outputs/t04/translations_haiti_test.jsonl"),
}


class FineTuned:
    """Laya's Agent behind the router's `predict(state, questions, model=...)` signature."""

    def __init__(self, path: str):
        import laya

        self.agent = laya.load(path, device="cuda")

    def predict(self, state, questions, model=None):
        res = self.agent.predict(state, questions)
        res["routing"] = {"model": "laya-ft"}
        return res


def rows(name: str, split: str) -> list[dict]:
    track, setting, _ = RUNS[name]
    col = SETTINGS[setting]["text"]
    df = dev_sample(track) if split == "dev" else test_split(track)
    if col == "text_mt":
        mt = {r["uid"]: r["text_mt"] for r in map(json.loads, MT[split].open())}
        df = df.assign(text_mt=df["uid"].map(mt))
    return [{"uid": u, "text": t} for u, t in zip(df["uid"], df[col], strict=True)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ft = FineTuned("data/laya_ft")
    timing = {}
    for name, (track, setting, _) in RUNS.items():
        qs = run_questions(track, SETTINGS[setting]["variant"])
        for split in ("dev", "test"):
            print(name, split, flush=True)
            run(ft, rows(name, split), qs, "text", None, OUT / f"{name}__{split}.jsonl")
        sample = rows(name, "test")[:110]
        times = []
        for k, r in enumerate(sample):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ft.predict({"message": r["text"]}, qs)
            torch.cuda.synchronize()
            if k >= 10:
                times.append((time.perf_counter() - t0) * 1e3)
        timing[name] = {"questions": len(qs), **summarize(times)}
        print(name, timing[name], flush=True)
    (OUT / "timing.json").write_text(
        json.dumps({"machine": machine_label(), "warmup": 10, "runs": timing}, indent=2)
    )


if __name__ == "__main__":
    main()
