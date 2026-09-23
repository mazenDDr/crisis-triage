"""T05: score Laya and every baseline on the same test messages, with paired differences.

Reads Laya's T04 answers and the baselines' T05 answers, scores them all with `evaluate`, and
writes results/t05_test.json: metrics per run and system, Laya minus each baseline (paired
bootstrap on the same messages), and speed and cost from the measured one-message timings.
"""

import json
from pathlib import Path

from crisis_triage.evaluate import ci, score_choice, score_multilabel
from crisis_triage.laya_run import load
from crisis_triage.runs import RUNS, SETTINGS

T05 = Path("outputs/t05")
BASELINES = ("e5lr", "qwen3-4b", "gemma3-4b")
FINE_TUNED = "laya-ft"  # T07: Laya fine-tuned on the dev splits
SYSTEMS = ("laya", FINE_TUNED, *BASELINES)
# GPU-hours are measured; dollars use this assumed rental price for one consumer GPU.
USD_PER_GPU_HOUR = 0.50


def paths(system: str, name: str) -> tuple[Path, Path | None]:
    """(test answers, dev answers) for one system and run."""
    dev_ref = RUNS[name][2]
    if system == "laya":
        return Path("outputs/t04") / f"{name}.jsonl", Path(dev_ref) if dev_ref else None
    base = Path("outputs/t07") if system == FINE_TUNED else T05 / system
    return base / f"{name}__test.jsonl", (base / f"{name}__dev.jsonl") if dev_ref else None


def timings() -> dict:
    out = {"laya": json.loads(Path("results/t04_latency.json").read_text())["rtx5060ti"]["runs"]}
    for system in (FINE_TUNED, *BASELINES):
        path = paths(system, "humaid")[0].parent / "timing.json"
        if path.exists():
            out[system] = json.loads(path.read_text())["runs"]
    return out


def main():
    results, paired = {}, {}
    for name, (track, setting, dev_ref) in RUNS.items():
        variant = SETTINGS[setting]["variant"]
        results[name], fns = {}, {}
        systems = [
            s
            for s in SYSTEMS
            if paths(s, name)[0].exists() and (not dev_ref or paths(s, name)[1].exists())
        ]
        # Every system is scored on the messages all of them answered (the LLM sample).
        common = set.intersection(*(set(load(paths(s, name)[0])) for s in systems))
        for system in systems:
            test, dev = paths(system, name)
            if dev_ref:
                results[name][system], fns[system] = score_multilabel(
                    track, variant, test, dev, common
                )
            else:
                results[name][system], fns[system] = score_choice(track, variant, test, common)
            print(name, system, "scored", flush=True)
        n = next(iter(results[name].values()))["n"]
        paired[name] = {
            f"{a} - {b}": ci(lambda i, a=a, b=b, f=fns: f[a](i) - f[b](i), n)
            for a in ("laya", FINE_TUNED)
            for b in fns
            if a in fns and b != a and not (a == FINE_TUNED and b == "laya")
        }
    speed = {}
    for system, runs in timings().items():
        speed[system] = {
            name: {
                "p50_ms": t["p50_ms"],
                "p95_ms": t["p95_ms"],
                "gpu_hours_per_1m": round(t["mean_ms"] * 1e6 / 3.6e6, 1),
                "usd_per_1m": round(t["mean_ms"] * 1e6 / 3.6e6 * USD_PER_GPU_HOUR, 2),
            }
            for name, t in runs.items()
            # Laya's haiti_mt time includes the NLLB step; the baselines read the saved
            # translations, so their haiti_mt times are not comparable and are left out.
            if system == "laya" or name != "haiti_mt"
        }
    out = {
        "metric_for_paired": "mean AUC (multi-label runs) or macro-F1 (choice runs)",
        "usd_per_gpu_hour_assumed": USD_PER_GPU_HOUR,
        "timing": "one message at a time on an RTX 5060 Ti; cost from the mean time. "
        "Laya's haiti_mt includes NLLB translation; baselines' haiti_mt is not timed.",
        "results": results,
        "paired": paired,
        "speed": speed,
    }
    Path("results/t05_test.json").write_text(json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
