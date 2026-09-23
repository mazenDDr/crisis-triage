"""Export the triage board's messages: a seeded sample of test tweets with every system's answer.

Only the two CC BY-NC-SA 4.0 tweet tracks (HumAID, CrisisBench) are shown; the Haiti SMS set
states no licence, so its text is not published. User names become @user, links are removed,
and phone numbers and email addresses are masked. Output: results/demo_messages.json
(CC BY-NC-SA 4.0, like its sources).
"""

import json
import re
from pathlib import Path

from crisis_triage import questions as Q
from crisis_triage.laya_run import load
from crisis_triage.runs import RUNS, SETTINGS, test_split

N = 150
TRACKS = ("humaid", "crisisbench_ml")
SYSTEMS = {
    "laya-ft": "outputs/t07/{run}__test.jsonl",
    "laya": "outputs/t04/{run}.jsonl",
    "e5lr": "outputs/t05/e5lr/{run}__test.jsonl",
    "qwen3-4b": "outputs/t05/qwen3-4b/{run}__test.jsonl",
    "gemma3-4b": "outputs/t05/gemma3-4b/{run}__test.jsonl",
}


# A lead byte (Â..ô) followed by continuation bytes, as they look when UTF-8 is read as cp1252.
MOJIBAKE = re.compile(
    "[\u00c2-\u00f4][\u0080-\u00bf\u0152-\u017e\u02c6\u02dc\u2013-\u203a\u20ac\u2122]+"
)


def repair(text: str) -> str:
    """Undo UTF-8 read as cp1252 ("caÃ­das" -> "caídas"), one broken sequence at a time, and
    only where the sequence decodes cleanly."""

    def fix(m: re.Match) -> str:
        try:
            return m.group().encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return m.group()

    return MOJIBAKE.sub(fix, text)


def clean(text: str) -> str:
    text = repair(text)
    text = re.sub(r"\bhttps?\S*|\bhttp…?", "", text)  # links, also ones cut short
    text = re.sub(r"\S+@\S+\.\w+", "[email]", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(
        r"\+?\d[\d\s().-]{5,}\d",
        lambda m: "[number]" if len(re.sub(r"\D", "", m.group())) >= 7 else m.group(),
        text,
    )
    text = re.sub(r"^RT @user:?\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    out = {"source_licence": "CC BY-NC-SA 4.0 (HumAID, CrisisBench)", "tracks": {}}
    for run in TRACKS:
        track, setting, _ = RUNS[run]
        variant = SETTINGS[setting]["variant"]
        answers = {s: load(Path(p.format(run=run))) for s, p in SYSTEMS.items()}
        common = set.intersection(*(set(a) for a in answers.values()))
        df = test_split(track)
        df = df[df["uid"].isin(common)].sample(N, random_state=0)
        classes = list(Q.COARSE)
        rows = []
        for r in df.to_dict("records"):
            row = {
                "text": clean(r["text"]),
                "lang": r["lang"],
                "event": r["event"],
                "gold": classes.index(Q.TO_COARSE[r["label"]]),
                "systems": {},
            }
            for s, recs in answers.items():
                probs = recs[r["uid"]]["answers"][f"{variant}/label"]["probs"]
                p = [probs[Q.readable(c)] for c in classes]
                best = max(range(len(p)), key=p.__getitem__)
                row["systems"][s] = [best, round(p[best], 4)]
                if s == "laya-ft":  # the full answer, for the "one message, one pass" section
                    row["laya_ft_probs"] = [round(x, 4) for x in p]
            rows.append(row)
        out["tracks"][run] = {"classes": [Q.readable(c) for c in classes], "messages": rows}
    Path("results/demo_messages.json").write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print({t: len(v["messages"]) for t, v in out["tracks"].items()})


if __name__ == "__main__":
    main()
