"""Merge outputs/latency_<machine>.json from each machine into results/t04_latency.json."""

import json
from pathlib import Path

merged = {}
for path in sorted(Path("outputs").glob("latency_*.json")):
    rec = json.loads(path.read_text())
    merged[rec["machine"]] = rec
Path("results/t04_latency.json").write_text(json.dumps(merged, indent=2) + "\n")
print({m: {r: v["p50_ms"] for r, v in rec["runs"].items()} for m, rec in merged.items()})
