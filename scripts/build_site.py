"""Build site/index.html (the story page + triage board) from the template and results/*.json.

Every number, tweet and answer on the page comes from the committed results: recorded
answers (demo_messages.json), gating (t06_gating.json), scores and speed (t05_test.json),
fine-tuning (t07_finetune.json) and data sizes (data_summary.json).
"""

import json
from pathlib import Path

RESULTS = Path("results")
SYSTEMS = [
    ("laya-ft", "Laya, fine-tuned", "Laya taught for 17 minutes on past disasters"),
    ("e5lr", "Small classifier", "e5 embeddings + logistic regression, trained on past disasters"),
    ("qwen3-4b", "Qwen3-4B (LLM)", "a 4-billion-parameter chat model, asked the same questions"),
    ("gemma3-4b", "Gemma-3-4B (LLM)", "a 4-billion-parameter chat model, asked the same questions"),
    ("laya", "Laya, out of the box", "Laya as released, no training on this task"),
]
TRACKS = {
    "humaid": (
        "English tweets",
        "HumAID: 9 disasters from 2018-19, all after the ones used for tuning",
    ),
    "crisisbench_ml": (
        "Tweets in 5 languages",
        "CrisisBench: Spanish, French, Italian, Portuguese and Tagalog tweets, 2013-15",
    ),
}
LANGS = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "tl": "Tagalog",
}
SHOWCASE_MAX_CHARS = 170


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def num(x):
    """Round, and turn NaN/inf into None so the page's JSON stays valid."""
    if x is None or x != x or x in (float("inf"), float("-inf")):
        return None
    return round(x, 4)


def gating_view(g: dict) -> dict:
    out = {"curve": [[p["coverage"], num(p["accuracy"])] for p in g["curve"]]}
    for target, v in g["at"].items():
        out[target] = {
            "cut": num(v["dev_cut"]),
            "coverage": num(v["coverage_point"]),
            "accuracy": num(v["accuracy_covered"]),
            "naive_coverage": num(v["naive"]["coverage"]),
            "naive_accuracy": num(v["naive"]["accuracy_covered"]),
        }
    return out


def showcase(demo: dict) -> list[dict]:
    """The first short sample tweet in each language, whatever the answer was."""
    picked, seen = [], set()
    for run in TRACKS:
        for m in demo["tracks"][run]["messages"]:
            if m["lang"] in seen or len(m["text"]) > SHOWCASE_MAX_CHARS:
                continue
            seen.add(m["lang"])
            picked.append({"track": run, "language": LANGS.get(m["lang"], m["lang"]), **m})
    return picked


def build() -> str:
    """The page as a string (main() writes it; the tests compare it with the committed file)."""
    demo, gating, scores = (
        load("demo_messages.json"),
        load("t06_gating.json"),
        load("t05_test.json"),
    )
    ft, sizes = load("t07_finetune.json"), load("data_summary.json")
    res = scores["results"]
    tracks = {}
    for run, (label, note) in TRACKS.items():
        g = gating["runs"][run]
        tracks[run] = {
            "label": label,
            "note": note,
            "n_test": g["n_test"],
            "classes": demo["tracks"][run]["classes"],
            "messages": demo["tracks"][run]["messages"],
            "systems": {
                key: {
                    "macro_f1": num(res[run][key]["macro_f1"]["value"]),
                    "ece": num(res[run][key]["ece"]),
                    **gating_view(g["systems"][key]),
                }
                for key, _, _ in SYSTEMS
            },
        }
    haiti = {
        "original": {k: num(v["mean_auc"]["value"]) for k, v in res["haiti_original"].items()},
        "translated": num(res["haiti_mt"]["laya"]["mean_auc"]["value"]),
        "human": num(res["haiti_human"]["laya"]["mean_auc"]["value"]),
    }
    speed = {k: v["humaid"] for k, v in scores["speed"].items() if "humaid" in v}
    ft_runs = ft["timing_runs"]
    data = {
        "systems": [{"key": k, "label": lab, "note": n} for k, lab, n in SYSTEMS],
        "tracks": tracks,
        "showcase": showcase(demo),
        "haiti": haiti,
        "speed": speed,
        "ft_speed_range": [
            min(r["runs"]["humaid"]["p50_ms"] for r in ft_runs),
            max(r["runs"]["humaid"]["p50_ms"] for r in ft_runs),
        ],
        "finetune": {
            "minutes": round(ft["epochs"][-1]["seconds"] / 60),
            "examples": ft["train_examples"],
        },
        "sizes": {k: v["dev"]["rows"] + v["test"]["rows"] for k, v in sizes.items()},
    }
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = Path("site/index-template.html").read_text()
    assert "/*__DATA__*/null" in template
    return template.replace("/*__DATA__*/null", blob)


def main():
    Path("site/index.html").write_text(build())
    print("wrote site/index.html")


if __name__ == "__main__":
    main()
