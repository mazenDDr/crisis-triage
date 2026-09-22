"""Run Laya over a table of messages and save every answer, resumably.

One JSON line per message: uid, the checkpoint the router used, and for each question the full
probabilities (choice), P(true) (noul) or expected level + probabilities (score). Nothing is
thrown away, so any metric can be recomputed later without the GPU.
"""

from __future__ import annotations

import json
from pathlib import Path


def compact(answer: dict) -> dict:
    if answer["type"] == "noul":
        return {"p": answer["noul"]}
    if answer["type"] == "choice":
        return {"probs": answer["probabilities"]}
    return {"score": answer["score"], "probs": answer["probabilities"]}


def run(router, rows, questions: dict, text_col: str, model: str | None, out: Path) -> Path:
    """Answer `questions` for each row (dicts with `uid` and `text_col`), appending to `out`.

    Rows already in `out` are skipped. `model=None` lets Laya's router choose the checkpoint.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["uid"] for line in out.open()}
    with out.open("a") as f:
        for row in rows:
            if row["uid"] in done:
                continue
            res = router.predict({"message": row[text_col]}, questions, model=model)
            rec = {
                "uid": row["uid"],
                "routed_to": res["routing"]["model"],
                "answers": {k: compact(a) for k, a in res["answers"].items()},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def load(path: Path) -> dict[str, dict]:
    return {r["uid"]: r for r in map(json.loads, path.open())}
