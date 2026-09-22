# crisis-triage

Sorting real disaster messages, in many languages, into a triage board with
[Laya](https://github.com/NandhaKishorM/laya): one forward pass answers what is needed, how
urgent it is, and whether a location is given, with a confidence you can act on.

Laya is measured against local LLMs on the same held-out disasters for accuracy, calibration,
speed and cost. Work in progress: no results yet.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
make check
```

GPU work (Laya on CUDA, LLM baselines) runs on a separate CUDA machine; see `gpu`.

## Licence

Code: MIT. Laya: Apache-2.0. Dataset licences are listed here once the data is chosen.
