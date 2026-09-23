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

## Data

Four tracks, built by `scripts/build_data.py` (counts in
[`results/data_summary.json`](results/data_summary.json)). Dev is used to tune questions and train
baselines; test is only used for reported numbers.

| Track | What | Languages | Dev / test | How it is split |
|---|---|---|---|---|
| `haiti_sms` | SMS sent to the 4636 line after the 2010 Haiti earthquake (a few from the 2010 Pakistan floods): original text + English translation, labelled needs | Haitian Creole, French (+ English) | 8,609 / 996 | official split (the set has no event column) |
| `humaid` | tweets from 19 disasters, 10 humanitarian classes | English | 37,211 / 39,265 | by time: 2016–17 events / 2018–19 events |
| `crisisbench_ml` | the non-English tweets of CrisisBench, 16 humanitarian classes | es, it, fr, tl, pt | 2,737 / 5,534 | by time: 2011–12 events / 2013–15 events |
| `humset` | humanitarian report excerpts, labelled sectors | en, fr, es | 131,495 / 14,571 | official split (documents separate, projects shared) |

Copies are removed: texts seen in dev are dropped from test, and CrisisBench rows that copy
Disaster Response messages are dropped. The Creole/French tag on `haiti_sms` is a word-list
heuristic (`unk` when neither list matches); a hand check of 30 messages per tag found 29/30
French and 25/30 Creole. No dataset labels urgency or location.

Not used: Kawarith (Arabic crisis tweets) ships tweet IDs only, without text.

## Results so far

Laya zero-shot on the test splits, with every setting chosen on dev:
[`results/t04_test.json`](results/t04_test.json) (scores with 95% bootstrap intervals and
calibration) and [`results/t04_latency.json`](results/t04_latency.json) (one message at a time
on an RTX 5060 Ti and an M4 Pro Mac). Laya against two local LLMs and a small trained
classifier, on the same test messages: [`results/t05_test.json`](results/t05_test.json). The
README tables will be generated from these files.

## Licences

| | Licence |
|---|---|
| Code | MIT |
| [Laya](https://github.com/NandhaKishorM/laya) | Apache-2.0 |
| [NLLB-200 distilled 600M / 1.3B](https://huggingface.co/facebook/nllb-200-distilled-600M) (translation of Creole/French SMS) | CC BY-NC 4.0 |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (baseline) | Apache-2.0 |
| [Gemma 3 4B IT](https://huggingface.co/google/gemma-3-4b-it) (baseline) | Gemma Terms of Use |
| [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) (baseline) | MIT |
| [HumAID](https://huggingface.co/datasets/QCRI/HumAID-events) | CC BY-NC-SA 4.0 |
| [CrisisBench](https://huggingface.co/datasets/QCRI/CrisisBench-all-lang) | CC BY-NC-SA 4.0 |
| [HumSet](https://huggingface.co/datasets/nlp-thedeep/humset) | Apache-2.0 |
| [Disaster Response Messages](https://huggingface.co/datasets/community-datasets/disaster_response_messages) (Appen) | not stated on the dataset card |

This repository does not redistribute any dataset text: data is downloaded by
`scripts/download_data.sh`, and committed results hold counts and scores only.
