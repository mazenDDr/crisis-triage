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

- **After 17 minutes of fine-tuning, Laya sends 38% of tweets from new disasters
  on its own, and 94.9% of those are right** (the target was 95%). The other
  62% go to a person.
- **Chat models are sure of almost everything.** Trusting Qwen3-4B's own "≥ 95% sure" sends
  95% of tweets on their own, but only 72% are right: about
  26 wrong answers per 100 tweets that nobody checks.
- **Out of the box, Laya was the weakest of five** (macro-F1 0.57). Fine-tuned on
  40,737 examples from earlier disasters, it is **tied for the best with a tuned e5 classifier** on the later ones
  (0.72; difference +0.009 [-0.004, +0.021], paired on the same 8,000 tweets), and
  its calibration error is the lowest (0.010; next: a tuned e5 classifier, 0.012).
- **It didn't read Haitian Creole well:** mean AUC 0.67 on the original SMS. Translating
  first (NLLB-600M) raises it to 0.77; fine-tuning, to 0.84.
- **The safety line can slip on new events:** on the multilingual tweets, the line chosen on
  earlier disasters reached 88.5% right on test, not 95%.

## At the desk: HumAID, 8,000 test tweets from 9 disasters

Every system answers the same 8-class question ("which kind of information does this crisis
message give?"). "Sent on its own" uses the lowest confidence cut that reached 95% right on
**dev** (earlier disasters), applied to test. "Trust its own" sends whenever the model says it is
at least 95% sure. Time is one message at a time on an RTX 5060 Ti.

| | Category score (macro-F1) | Sent on its own at the 95% target | …right among those | Trust its own “≥ 95% sure” | …right among those | Time per message |
|---|---|---|---|---|---|---|
| **Laya, fine-tuned** | 0.725 [0.712, 0.737] | 38% | 94.9% | 22% | 98.4% | 46–53 ms |
| e5 + LR (trained, tuned on dev) | 0.716 [0.703, 0.728] | 33% | 95.4% | 22% | 97.4% | 6 ms |
| Qwen3-4B | 0.642 [0.630, 0.653] | 0% | — | 95% | 72.2% | 114 ms |
| Gemma-3-4B | 0.597 [0.584, 0.609] | 0% | — | 96% | 66.7% | 126 ms |
| Laya, out of the box | 0.568 [0.556, 0.579] | 6% | 80.5% | 0% | — | 24 ms |

"—": nothing sent. A 95% interval is shown for the category score; every other interval is in
[`results/t06_gating.json`](results/t06_gating.json).

## Every test set

Same test messages for every system (the LLMs answered a fixed random sample of the two largest
sets). Bold marks the best score in each row. Intervals and paired differences are in
[`results/t05_test.json`](results/t05_test.json).

| Test set | n | Metric | **Laya, fine-tuned** | e5 + LR (trained, tuned on dev) | Qwen3-4B | Gemma-3-4B | Laya, out of the box |
|---|---|---|---|---|---|---|---|
| Haiti SMS, Creole/French original | 996 | mean AUC | 0.838 | **0.873** | 0.762 | 0.813 | 0.669 |
| Haiti SMS, NLLB translation | 996 | mean AUC | 0.844 | — | 0.822 | **0.851** | 0.772 |
| Haiti SMS, human translation | 996 | mean AUC | 0.905 | — | 0.879 | **0.927** | 0.849 |
| HumAID tweets, 2018–19 disasters | 8,000 | macro-F1 | **0.725** | 0.716 | 0.642 | 0.597 | 0.568 |
| CrisisBench tweets, es/fr/it/pt/tl | 5,534 | macro-F1 | **0.453** | 0.410 | 0.426 | 0.387 | 0.280 |
| HumSet report excerpts | 3,000 | mean AUC | 0.934 | **0.956** | 0.932 | 0.910 | 0.691 |

- **Laya, out of the box**: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), no training on this task, with question wording and checkpoint chosen on dev.
- **Laya, fine-tuned**: one multilingual checkpoint trained on every track's dev split at once ([`scripts/finetune_laya.py`](scripts/finetune_laya.py)).
- **e5 + LR**: multilingual-e5-base embeddings and logistic regression, trained on the same dev data; regularisation and class weighting chosen by cross-validation on dev.
- **Qwen3-4B, Gemma-3-4B**: zero-shot, asked Laya's exact questions and read out from next-token probabilities, so nothing is parsed.

## How many labels is Laya worth?

Laya out of the box needs no labels. A small classifier (e5 + logistic regression) needs some.
Here it is trained on N random dev messages (5 draws each) and scored on **every** test
message (39,265 English, 5,534 multilingual); "beats" counts the draws whose paired 95% interval against Laya out of the box is
entirely above 0. "Tuned" uses the setting chosen by cross-validation on the full dev split,
which slightly favours the small budgets.

| Labelled messages | English tweets (default) | English tweets (tuned) | Multilingual tweets (default) | Multilingual tweets (tuned) |
|---|---|---|---|---|
| 25 | 0.09 | 0.18 | 0.07 | 0.10 |
| 50 | 0.09 | 0.22 | 0.07 | 0.17 |
| 100 | 0.14 | 0.35 | 0.07 | 0.23 |
| 200 | 0.16 | 0.44 | 0.07 | 0.29 **beats 2/5** |
| 400 | 0.29 | 0.56 **beats 2/5** | 0.09 | 0.35 **beats 5/5** |
| 800 | 0.37 | 0.60 **beats 4/5** | 0.10 | 0.37 **beats 5/5** |
| 1,600 | 0.47 | 0.65 **beats 5/5** | 0.21 | 0.39 **beats 5/5** |
| 2,737 (all dev) | — | — | 0.29 | 0.41 **beats 1/1** |
| 3,200 | 0.58 **beats 3/5** | 0.68 **beats 5/5** | — | — |
| 6,400 | 0.63 **beats 5/5** | 0.70 **beats 5/5** | — | — |
| 37,211 (all dev) | 0.70 **beats 1/1** | 0.72 **beats 1/1** | — | — |
| *Laya out of the box* | *0.56* | | *0.28* | |
| *Laya fine-tuned* | *0.73* | | *0.45* | |

With tuned settings, every draw beats Laya out of the box from **1,600** labelled English
tweets and **400** multilingual ones. No draw at any budget beats the fine-tuned Laya; trained on all of dev, the classifier is significantly behind it on 2 of 2 tracks. With scikit-learn's default settings the classifier needs thousands, which is
why the first version of this comparison under-rated it; the baseline is now tuned on dev
([`trained.py`](src/crisis_triage/trained.py)).

## Languages: the Haitian Creole SMS

| Haiti SMS, 996 test messages | Mean AUC over 4 needs [95% CI] |
|---|---|
| Laya, out of the box, reads the Creole | 0.669 [0.625, 0.713] |
| Laya, after an NLLB-600M translation to English | 0.772 [0.731, 0.809] |
| Laya, fine-tuned, reads the Creole | 0.838 [0.800, 0.879] |
| Laya with a human translator (upper bound) | 0.849 [0.820, 0.878] |

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
