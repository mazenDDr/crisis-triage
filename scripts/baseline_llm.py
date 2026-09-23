"""T05 LLM baselines: the chosen questions of every run, asked to a local LLM on gpu-box.

`--model qwen3-4b|gemma3-4b`: answers for the fixed dev sample (thresholds) and every test
message, in Laya's format, at outputs/t05/<model>/<run>__<split>.jsonl (resumable); then
one-message-at-a-time timing (all of a message's questions in one batch, like Laya's call).
`--check`: only the first 100 dev messages of each run, to catch prompt or token problems.
"""

import argparse
import json
import time
from pathlib import Path

from crisis_triage.latency import machine_label, summarize
from crisis_triage.laya_run import load
from crisis_triage.llm import LLMJudge
from crisis_triage.runs import RUNS, SETTINGS, llm_test_split, run_questions
from crisis_triage.scoring import dev_sample

MODELS = {"qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507", "gemma3-4b": "google/gemma-3-4b-it"}
MT = {
    "dev": Path("outputs/t03b/mt_nllb600m_lower.jsonl"),
    "test": Path("outputs/t04/translations_haiti_test.jsonl"),
}
CHUNK = 256  # messages per write


def rows(name: str, split: str) -> list[dict]:
    track, setting, _ = RUNS[name]
    col = SETTINGS[setting]["text"]
    df = dev_sample(track) if split == "dev" else llm_test_split(track)
    if col == "text_mt":
        mt = {r["uid"]: r["text_mt"] for r in map(json.loads, MT[split].open())}
        df = df.assign(text_mt=df["uid"].map(mt))
    return [{"uid": u, "text": t} for u, t in zip(df["uid"], df[col], strict=True)]


def answer_rows(judge: LLMJudge, qs: dict, todo: list[dict]) -> list[dict]:
    items = [(k, q, r["text"]) for r in todo for k, q in qs.items()]
    answers = iter(judge.answer(items))
    return [{"uid": r["uid"], "answers": {k: next(answers) for k in qs}} for r in todo]


def run_model(model: str, check: bool):
    import torch

    judge = LLMJudge(MODELS[model])
    out = Path("outputs/t05") / model
    out.mkdir(parents=True, exist_ok=True)
    for split in ("dev",) if check else ("dev", "test"):
        for name, (track, setting, _) in RUNS.items():
            qs = run_questions(track, SETTINGS[setting]["variant"])
            path = out / f"{name}__{split}{'_check' if check else ''}.jsonl"
            done = set(load(path)) if path.exists() else set()
            todo = [r for r in rows(name, split) if r["uid"] not in done]
            if check:
                todo = todo[:100]
            print(model, name, split, len(todo), flush=True)
            with path.open("a") as f:
                for i in range(0, len(todo), CHUNK):
                    for rec in answer_rows(judge, qs, todo[i : i + CHUNK]):
                        rec["routed_to"] = model
                        f.write(json.dumps(rec) + "\n")
    if check:
        return
    timing = {}
    for name, (track, setting, _) in RUNS.items():
        qs = run_questions(track, SETTINGS[setting]["variant"])
        sample = rows(name, "test")[:110]
        times = []
        for k, r in enumerate(sample):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            answer_rows(judge, qs, [r])
            torch.cuda.synchronize()
            if k >= 10:
                times.append((time.perf_counter() - t0) * 1e3)
        timing[name] = {"questions": len(qs), **summarize(times)}
        print(name, timing[name], flush=True)
    (out / "timing.json").write_text(
        json.dumps({"machine": machine_label(), "warmup": 10, "runs": timing}, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run_model(args.model, args.check)
