"""T04: Laya zero-shot on the test splits, with every setting chosen on dev (T03, T03b).

`--run` (gpu-box): translate the Haiti test SMS with the chosen NLLB setting, then answer the
chosen questions for every test message. `--score`: metrics with 95% bootstrap intervals,
calibration, and thresholds taken from the dev answers. Results: results/t04_test.json.
"""

import argparse
import json
from pathlib import Path

from crisis_triage.evaluate import ci, score_choice, score_multilabel
from crisis_triage.laya_run import run
from crisis_triage.runs import RUNS, SETTINGS, run_questions, test_split
from crisis_triage.translate import CHOSEN_MT, NLLB_CODES, Translator

OUT = Path("outputs/t04")


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


def score_all() -> dict:
    results, row_fns = {}, {}
    for name, (track, setting, dev_path) in RUNS.items():
        path = OUT / f"{name}.jsonl"
        if not path.exists():
            continue
        variant = SETTINGS[setting]["variant"]
        if dev_path:
            results[name], row_fns[name] = score_multilabel(track, variant, path, Path(dev_path))
        else:
            results[name], row_fns[name] = score_choice(track, variant, path)
        print(name, "scored", flush=True)
    n = results["haiti_mt"]["n"]
    results["haiti_paired_mean_auc"] = {
        "mt - original": ci(lambda i: row_fns["haiti_mt"](i) - row_fns["haiti_original"](i), n),
        "human - mt": ci(lambda i: row_fns["haiti_human"](i) - row_fns["haiti_mt"](i), n),
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
