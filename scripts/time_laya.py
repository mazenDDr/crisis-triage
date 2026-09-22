"""Speed of each T04 run, one message at a time, on the GPU machine and on the Mac.

`--export` (gpu-box): pick 110 test messages per run into data/timing_sample.jsonl (copy that
file to the Mac's data/ for the Mac run). `--device cuda|mps`: time Laya's predict with the
run's exact questions and checkpoint: 10 warm-up messages, then 100 timed. On CUDA the Haiti
MT run is timed end to end (NLLB translation + Laya). Output: outputs/latency_<machine>.json.
"""

import argparse
import json
import time
from pathlib import Path

from crisis_triage.latency import machine_label, summarize
from crisis_triage.runs import RUNS, SETTINGS, run_questions, test_split
from crisis_triage.translate import CHOSEN_MT, NLLB_CODES, Translator

SAMPLE = Path("data/timing_sample.jsonl")
WARMUP, TIMED = 10, 100


def export():
    with SAMPLE.open("w") as f:
        for name, (track, setting, _) in RUNS.items():
            col = SETTINGS[setting]["text"]
            df = test_split(track).sample(WARMUP + TIMED, random_state=0)
            for r in df.to_dict("records"):
                text = r["text"] if col == "text_mt" else r[col]  # MT runs translate live
                rec = {"run": name, "uid": r["uid"], "text": text, "lang": r["lang"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sync(device: str):
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_runs(device: str) -> dict:
    from laya import Router

    rows = [json.loads(line) for line in SAMPLE.open()]
    router = Router(preload=True, device=device)  # no reloads on checkpoint switches
    translator = None
    out = {"machine": machine_label(), "device": device, "warmup": WARMUP, "runs": {}}
    for name, (track, setting, _) in RUNS.items():
        cfg = SETTINGS[setting]
        mt = cfg["text"] == "text_mt"
        if mt and device != "cuda":
            continue  # translation is timed on the GPU machine only
        if mt and translator is None:
            translator = Translator(CHOSEN_MT["model"], num_beams=CHOSEN_MT["num_beams"])
        qs = run_questions(track, cfg["variant"])
        mine = [r for r in rows if r["run"] == name]
        times = []
        for k, r in enumerate(mine):
            sync(device)
            t0 = time.perf_counter()
            text = r["text"]
            if mt:
                text = translator([text], NLLB_CODES[r["lang"]], CHOSEN_MT["lowercase"])[0]
            router.predict({"message": text}, qs, model=cfg["model"])
            sync(device)
            if k >= WARMUP:
                times.append((time.perf_counter() - t0) * 1e3)
        out["runs"][name] = {"questions": len(qs), "timed": len(times), **summarize(times)}
        print(name, out["runs"][name], flush=True)
    path = Path("outputs") / f"latency_{out['machine']}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()
    if args.export:
        export()
    if args.device:
        time_runs(args.device)
