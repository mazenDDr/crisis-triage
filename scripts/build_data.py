"""Build data/processed/<track>.parquet on the GPU machine and save the counts to results/."""

import json
from pathlib import Path

from crisis_triage.data import build

summary = build()
out = Path("results/data_summary.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
