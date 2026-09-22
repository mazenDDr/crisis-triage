#!/usr/bin/env bash
# Download every dataset into data/raw/ on the GPU machine (run with `./gpu run`).
set -euo pipefail
# One --include per pattern: extra patterns after a single --include are silently dropped.
dl() { hf download "$1" --repo-type dataset --local-dir "data/raw/$2" "${@:3}"; }

dl community-datasets/disaster_response_messages disaster_response
# HumAID-events: one folder per event (HumAID-all has the same tweets without the event).
dl QCRI/HumAID-events humaid
dl QCRI/CrisisBench-all-lang crisisbench --include "humanitarian/*" --include README.md
dl nlp-thedeep/humset humset \
  --include data/train.jsonl --include data/validation.jsonl --include data/test.jsonl --include README.md
# Not used: arbml/kawarith_* (Arabic) ships tweet IDs only, no text.
du -sh data/raw/*
