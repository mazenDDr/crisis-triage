"""Turn the raw datasets into one table per track, with a dev/test split that cannot leak.

Tracks:
- `haiti_sms`: Disaster Response "direct" SMS (mostly the 2010 Haiti earthquake, a few from the
  2010 Pakistan floods), the original text and its English translation, with multi-label needs.
- `humaid`: English tweets from 19 disasters (2016-2019), one humanitarian class each.
- `crisisbench_ml`: the non-English tweets of CrisisBench (es, it, fr, tl, pt), one class each.
- `humset`: humanitarian report excerpts (en, es, fr) with multi-label sectors.

The split is by time wherever there is more than one event: dev = past disasters (tune
questions, train baselines), test = later disasters, never seen while tuning. The SMS set has no
event column, so it keeps the official message split; HumSet keeps its official split, which
separates documents but shares projects. Both limits are stated wherever their numbers are
reported.

Texts that appear in dev are removed from test (retweets and copies), and CrisisBench rows that are
copies of Disaster Response messages are dropped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

HUMAID_DEV_UNTIL = 2017  # 2016-2017 events tune, 2018-2019 events test
CRISISBENCH_DEV_UNTIL = 2012  # 2011-2012 events tune, 2013-2015 events test
CRISISBENCH_LANGS = ("es", "it", "fr", "tl", "pt")

# Disaster Response columns grouped into the needs Laya is asked about.
HAITI_NEEDS = {
    "rescue": ("search_and_rescue", "missing_people"),
    "medical": ("medical_help", "medical_products"),
    "water_food": ("water", "food"),
    "shelter": ("shelter",),
}

CREOLE_WORDS = {"mwen", "nou", "yo", "pa", "ki", "nan", "bezwen", "gen", "pou", "kote", "manje"}
FRENCH_WORDS = {"je", "nous", "les", "des", "est", "une", "pour", "avons", "sont"}


def norm(text: str) -> str:
    """Lower case, no punctuation, no RT prefix, no links: used to find copies."""
    text = re.sub(r"https?://\S+|^rt @\w+:?", " ", str(text).lower())
    return re.sub(r"\W+", " ", text).strip()


def event_year(event: str) -> int | None:
    """First year in an event name such as `2014_chile_earthquake_esp` or `kerala_floods_2018`."""
    match = re.search(r"(?<!\d)(20\d\d)(?!\d)", event)
    return int(match.group(1)) if match else None


def time_split(year: int | None, dev_until: int) -> str | None:
    if year is None:
        return None
    return "dev" if year <= dev_until else "test"


def guess_creole_or_french(text: str) -> str:
    """`ht`, `fr` or `unk` for an original message, from common function words. A heuristic,
    not a language detector: it separates the two main languages of this dataset and leaves the
    rest (English, romanized Urdu, noise) as `unk`. "la" and "le" are left out because Creole
    uses them too."""
    words = set(re.findall(r"[a-zà-ÿ]+", str(text).lower()))
    ht, fr = len(words & CREOLE_WORDS), len(words & FRENCH_WORDS)
    if ht == fr == 0:
        return "unk"
    return "fr" if fr > ht else "ht"


def drop_test_copies(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate texts within a split, then test texts that also appear in dev."""
    key = df["text"].map(norm)
    df = df[~(key.duplicated() & key.ne(""))]
    key = key.loc[df.index]
    seen = set(key[df["split"] == "dev"])
    return df[~((df["split"] == "test") & key.isin(seen)) & key.ne("")]


def load_haiti() -> pd.DataFrame:
    parts = []
    for f in sorted((RAW / "disaster_response/data").glob("*.parquet")):
        d = pd.read_parquet(f)
        d["official"] = f.name.split("-")[0]
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d = d[(d["genre"] == "direct") & d["original"].fillna("").str.strip().ne("")]
    needs = {k: d[list(cols)].eq(1).any(axis=1) for k, cols in HAITI_NEEDS.items()}
    return pd.DataFrame(
        {
            "uid": "haiti-" + d.index.astype(str),
            "track": "haiti_sms",
            "source": "disaster_response",
            "event": "2010_direct_sms",
            "lang": d["original"].map(guess_creole_or_french),
            "text": d["original"].str.strip(),
            "text_en": d["message"].str.strip(),
            "labels": [
                [k for k in HAITI_NEEDS if needs[k].loc[i]] for i in d.index
            ],  # empty list = no listed need
            "request": d["request"].eq(1),
            "split": d["official"].map({"train": "dev", "validation": "dev", "test": "test"}),
        }
    )


def load_humaid() -> pd.DataFrame:
    parts = []
    for f in sorted((RAW / "humaid").glob("*/*.json")):
        d = pd.read_json(f, dtype={"tweet_id": str})
        d["event"] = f.parent.name
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    return pd.DataFrame(
        {
            "uid": "humaid-" + d["tweet_id"],
            "track": "humaid",
            "source": "humaid",
            "event": d["event"],
            "lang": "en",
            "text": d["tweet_text"],
            "label": d["class_label"],
            "split": d["event"].map(lambda e: time_split(event_year(e), HUMAID_DEV_UNTIL)),
        }
    )


def load_crisisbench(haiti_texts: set[str]) -> pd.DataFrame:
    d = pd.concat(
        [
            pd.read_json(f, dtype={"id": str})
            for f in (RAW / "crisisbench/humanitarian").glob("*.json")
        ],
        ignore_index=True,
    )
    d = d[d["lang"].isin(CRISISBENCH_LANGS) & (d["source"] != "drd-figureeight-multimedia")]
    d = d[~d["text"].map(norm).isin(haiti_texts)]
    return pd.DataFrame(
        {
            "uid": "cb-" + d["id"],
            "track": "crisisbench_ml",
            "source": "crisisbench/" + d["source"],
            "event": d["event"],
            "lang": d["lang"],
            "text": d["text"],
            "label": d["class_label"],
            "split": d["event"].map(lambda e: time_split(event_year(e), CRISISBENCH_DEV_UNTIL)),
        }
    )


def load_humset() -> pd.DataFrame:
    parts = []
    for name, split in (("train", "dev"), ("validation", "dev"), ("test", "test")):
        d = pd.read_json(RAW / f"humset/data/{name}.jsonl", lines=True)
        d["split"] = split
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    return pd.DataFrame(
        {
            "uid": "humset-" + d["entry_id"].astype(str),
            "track": "humset",
            "source": "humset",
            "event": "project-" + d["project_id"].astype(str),
            "lang": d["lang"],
            "text": d["excerpt"],
            "labels": d["sectors"],
            "split": d["split"],
        }
    )


def summarize(df: pd.DataFrame) -> dict:
    """Counts only (no text), safe to commit."""
    out = {"rows": int(len(df))}
    for split, part in df.groupby("split"):
        s = {
            "rows": int(len(part)),
            "events": int(part["event"].nunique()),
            "langs": part["lang"].value_counts().to_dict(),
        }
        if "label" in part and part["label"].notna().any():
            s["labels"] = part["label"].value_counts().to_dict()
        if "labels" in part and part["labels"].notna().any():
            s["labels"] = part["labels"].explode().value_counts().to_dict()
            s["no_label"] = int(part["labels"].map(len).eq(0).sum())
        out[split] = s
    return out


def build(out_dir: Path = PROCESSED) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    haiti = load_haiti()
    haiti_en = set(haiti["text_en"].map(norm))
    tracks = {
        "haiti_sms": haiti,
        "humaid": load_humaid(),
        "crisisbench_ml": load_crisisbench(haiti_en),
        "humset": load_humset(),
    }
    summary = {}
    for name, df in tracks.items():
        before = len(df)
        df = drop_test_copies(df[df["split"].notna()])
        df.reset_index(drop=True).to_parquet(out_dir / f"{name}.parquet")
        summary[name] = {"dropped_copies_or_unsplit": before - len(df), **summarize(df)}
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False))
