"""Build site/index.html (the triage board) from site/index-template.html and results/*.json.

Every number and message on the page comes from the committed results: the recorded answers
in results/demo_messages.json, gating in results/t06_gating.json, scores in t05_test.json.
"""

import json
from pathlib import Path

RESULTS = Path("results")
SYSTEMS = [
    ("laya-ft", "Laya fine-tuned", "Laya's multilingual model, fine-tuned on past disasters"),
    (
        "e5lr",
        "e5 + LR",
        "small trained classifier: multilingual-e5 embeddings + logistic regression",
    ),
    ("qwen3-4b", "Qwen3-4B", "4B-parameter LLM, zero-shot, same questions"),
    ("gemma3-4b", "Gemma-3-4B", "4B-parameter LLM, zero-shot, same questions"),
    ("laya", "Laya zero-shot", "Laya as released, no training on this task"),
]
TRACKS = {
    "humaid": (
        "English tweets",
        "HumAID: 9 disasters from 2018-19, after every event used for tuning",
    ),
    "crisisbench_ml": (
        "Tweets in 5 languages",
        "CrisisBench: Spanish, French, Italian, Portuguese and Tagalog tweets from 2013-15 events",
    ),
}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def gating_view(g: dict) -> dict:
    """The parts of one system's gating result the page needs."""
    out = {"curve": [[p["coverage"], round(p["accuracy"], 4)] for p in g["curve"]]}
    for target, v in g["at"].items():
        cut = v["dev_cut"]
        out[target] = {
            "cut": None if cut == float("inf") else round(cut, 4),
            "coverage": round(v["coverage_point"], 4),
            "accuracy": None
            if v["accuracy_covered"] != v["accuracy_covered"]
            else round(v["accuracy_covered"], 4),
            "naive_coverage": round(v["naive"]["coverage"], 4),
            "naive_accuracy": None
            if v["naive"]["accuracy_covered"] != v["naive"]["accuracy_covered"]
            else round(v["naive"]["accuracy_covered"], 4),
        }
    return out


def build() -> str:
    """The page as a string (main() writes it; the tests compare it with the committed file)."""
    demo, gating, scores = (
        load("demo_messages.json"),
        load("t06_gating.json"),
        load("t05_test.json"),
    )
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
                    "macro_f1": round(scores["results"][run][key]["macro_f1"]["value"], 3),
                    **gating_view(g["systems"][key]),
                }
                for key, _, _ in SYSTEMS
            },
        }
    data = {
        "systems": [{"key": k, "label": lab, "note": n} for k, lab, n in SYSTEMS],
        "tracks": tracks,
        "targets": gating["targets"],
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
