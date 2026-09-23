# crisis-triage

**Sure → sent. Unsure → a person.** After a disaster, messages arrive faster than anyone can
read them. This project measures whether a small model, [Laya](https://github.com/NandhaKishorM/laya),
can read crisis messages in many languages, answer typed questions about them in one pass, and,
most of all, **know when it isn't sure**, so that only its sure answers skip a person.

<p align="center"><img src="docs/assets/desk.svg" width="100%" alt="A real test tweet is read and stamped SENT because the model is sure; the next one falls below the safety line and is stamped FOR A PERSON."></p>

**See it:** [mazenddr.github.io/crisis-triage](https://mazenddr.github.io/crisis-triage/), where
real test tweets are sorted live from recorded answers. **Model:**
[mazenDDr/laya-crisis-triage](https://huggingface.co/mazenDDr/laya-crisis-triage).

## What we found

- **After {{ft_minutes}} minutes of fine-tuning, Laya sends {{ft_sent}} of tweets from new disasters
  on its own, and {{ft_right}} of those are right** (the target was 95%). The other
  {{ft_person}} go to a person.
- **Chat models are sure of almost everything.** Trusting Qwen3-4B's own "≥ 95% sure" sends
  {{qwen_sent}} of tweets on their own, but only {{qwen_right}} are right: about
  {{qwen_wrong_per_100}} wrong answers per 100 tweets that nobody checks.
- **Out of the box, Laya was the weakest of five systems** (macro-F1 {{zs_f1}}). Fine-tuned on
  {{ft_examples}} examples from earlier disasters, it is the best on the later ones
  ({{ft_f1}}; {{ft_vs_e5}} over a trained e5 classifier, paired on the same tweets), and its
  confidence is the most trustworthy (lowest calibration error).
- **It didn't read Haitian Creole well:** mean AUC {{haiti_zs}} on the original SMS. Translating
  first (NLLB-600M) raises it to {{haiti_mt}}; fine-tuning, to {{haiti_ft}}.
- **The safety line can slip on new events:** on the multilingual tweets, the line chosen on
  earlier disasters reached {{cb_reached}} right on test, not 95%.

## At the desk: HumAID, {{n_humaid}} test tweets from 9 disasters

Every system answers the same 8-class question ("which kind of information does this crisis
message give?"). "Sent on its own" uses the lowest confidence cut that reached 95% right on
**dev** (earlier disasters), applied to test. "Trust its own" sends whenever the model says it is
at least 95% sure. Time is one message at a time on an RTX 5060 Ti.

{{desk_table}}

"—": nothing sent. A 95% interval is shown for the category score; every other interval is in
[`results/t06_gating.json`](results/t06_gating.json).

## Every test set

Same test messages for every system (the LLMs answered a fixed random sample of the two largest
sets). Bold marks the best score in each row. Intervals and paired differences are in
[`results/t05_test.json`](results/t05_test.json).

{{all_runs_table}}

- **Laya, out of the box**: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), no training on this task, with question wording and checkpoint chosen on dev.
- **Laya, fine-tuned**: one multilingual checkpoint trained on every track's dev split at once ([`scripts/finetune_laya.py`](scripts/finetune_laya.py)).
- **e5 + LR**: multilingual-e5-base embeddings and logistic regression, trained on the same dev data.
- **Qwen3-4B, Gemma-3-4B**: zero-shot, asked Laya's exact questions and read out from next-token probabilities, so nothing is parsed.

## Languages: the Haitian Creole SMS

{{creole_table}}

Mean AUC over rescue, medical, water/food and shelter needs. The Haiti set is one event with a
message-level split, so it favours trained models; HumAID and CrisisBench are split by time.

## How it was measured

1. **Data** ([`src/crisis_triage/data.py`](src/crisis_triage/data.py)): four public collections,
   split so that nothing is tuned on the disasters it is tested on (below).
2. **Questions** ([`questions.py`](src/crisis_triage/questions.py)): wording, class set and
   checkpoint chosen on a 1,000-message dev sample per track, each choice with a paired bootstrap
   interval.
3. **Scoring** ([`evaluate.py`](src/crisis_triage/evaluate.py)): one scorer for every system;
   95% bootstrap intervals; paired differences on the same messages; ECE and Brier for calibration.
4. **Gating** ([`gating.py`](src/crisis_triage/gating.py)): the confidence cut is picked on dev and
   checked on test; tied confidences are kept or dropped together.
5. **Speed**: one message at a time; the fine-tuned model's time varied between runs on this WSL
   GPU, so a range is reported.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
make check      # lint, format, tests (the README and the page must match a fresh build)
make readme     # this README and docs/assets/desk.svg, from results/*.json
make site       # site/index.html, from results/*.json
```

GPU steps (downloads, Laya, LLM baselines, fine-tuning) run on a CUDA machine with the scripts in
[`scripts/`](scripts/); data lands in `data/` and raw answers in `outputs/`, neither committed.

## Data

Four tracks, built by `scripts/build_data.py` (counts in
[`results/data_summary.json`](results/data_summary.json)). Dev tunes questions and trains
baselines; test is only used for reported numbers.

| Track | What | Languages | Dev / test | How it is split |
|---|---|---|---|---|
| `haiti_sms` | SMS sent to the 4636 line after the 2010 Haiti earthquake (a few from the 2010 Pakistan floods): original text + English translation, labelled needs | Haitian Creole, French (+ English) | 8,609 / 996 | official split (the set has no event column) |
| `humaid` | tweets from 19 disasters, 10 humanitarian classes | English | 37,211 / 39,265 | by time: 2016–17 events / 2018–19 events |
| `crisisbench_ml` | the non-English tweets of CrisisBench, 16 humanitarian classes | es, it, fr, tl, pt | 2,737 / 5,534 | by time: 2011–12 events / 2013–15 events |
| `humset` | humanitarian report excerpts, labelled sectors | en, fr, es | 131,495 / 14,571 | official split (documents separate, projects shared) |

Copies are removed: texts seen in dev are dropped from test, and CrisisBench rows that copy
Disaster Response messages are dropped. No dataset labels urgency, so urgency is only checked
against a stand-in (rescue/medical needs; injured, missing or requests); location is not scored.
Not used: Kawarith (Arabic crisis tweets) ships tweet IDs only, without text.

## Limits

- Recorded, not deployed: nothing here ran in a real emergency. Keep a person in the loop.
- The chat models compared are 4B models run locally; larger paid models were not tested.
- Fine-tuning made the untrained urgency question slightly worse on the Haiti SMS.
- Only 11 Haiti test messages ask for rescue, so that need's score is noisy.

## Licences

| | Licence |
|---|---|
| Code | MIT |
| [Laya](https://github.com/NandhaKishorM/laya) | Apache-2.0 |
| [Fine-tuned model](https://huggingface.co/mazenDDr/laya-crisis-triage) | CC BY-NC-SA 4.0 |
| [NLLB-200 distilled 600M / 1.3B](https://huggingface.co/facebook/nllb-200-distilled-600M) (translation of Creole/French SMS) | CC BY-NC 4.0 |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (baseline) | Apache-2.0 |
| [Gemma 3 4B IT](https://huggingface.co/google/gemma-3-4b-it) (baseline) | Gemma Terms of Use |
| [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) (baseline) | MIT |
| [HumAID](https://huggingface.co/datasets/QCRI/HumAID-events) | CC BY-NC-SA 4.0 |
| [CrisisBench](https://huggingface.co/datasets/QCRI/CrisisBench-all-lang) | CC BY-NC-SA 4.0 |
| [HumSet](https://huggingface.co/datasets/nlp-thedeep/humset) | Apache-2.0 |
| [Disaster Response Messages](https://huggingface.co/datasets/community-datasets/disaster_response_messages) (Appen) | not stated on the dataset card |

`results/demo_messages.json` (the 300 tweets on the page, user names, links and phone numbers
masked) is CC BY-NC-SA 4.0, like HumAID and CrisisBench. Apart from that file, this repository
does not redistribute dataset text: data is downloaded by `scripts/download_data.sh`, and
committed results hold counts and scores only.
