"""Write the Hugging Face model card for the fine-tuned Laya from the committed results.

Numbers come from results/t05_test.json and results/t07_finetune.json; the example questions
come from questions.py, so the card shows exactly what the model was trained and tested on.

  python scripts/build_model_card.py data/laya_ft/README.md
"""

import json
import sys
from pathlib import Path

from crisis_triage import questions as Q
from crisis_triage.runs import run_questions

REPO = "https://github.com/mazenDDr/crisis-triage"
RUN_NAMES = {
    "haiti_original": "Haiti SMS, Creole/French original",
    "haiti_mt": "Haiti SMS, NLLB-600M translation",
    "haiti_human": "Haiti SMS, human translation",
    "humaid": "HumAID tweets, 2018-19 disasters",
    "crisisbench_ml": "CrisisBench tweets, es/fr/it/pt/tl",
    "humset": "HumSet report excerpts, en/fr/es",
}
SYSTEMS = {
    "laya-ft": "**this model**",
    "laya": "Laya zero-shot",
    "e5lr": "e5 + LR",
    "qwen3-4b": "Qwen3-4B",
    "gemma3-4b": "Gemma-3-4B",
}


def main_metric(r: dict) -> float:
    return (r.get("mean_auc") or r["macro_f1"])["value"]


def results_table(res: dict) -> str:
    head = "| Test set | n | Metric | " + " | ".join(SYSTEMS.values()) + " |"
    rows = [head, "|" + "---|" * (3 + len(SYSTEMS))]
    for run, label in RUN_NAMES.items():
        systems = res["results"][run]
        first = next(iter(systems.values()))
        metric = "mean AUC" if "mean_auc" in first else "macro-F1"
        cells = [f"{main_metric(systems[s]):.3f}" if s in systems else "—" for s in SYSTEMS]
        rows.append(f"| {label} | {first['n']:,} | {metric} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def ece_table(res: dict) -> str:
    rows = [
        "| Test set | " + " | ".join(SYSTEMS.values()) + " |",
        "|" + "---|" * (1 + len(SYSTEMS)),
    ]
    for run in ("haiti_original", "humaid", "crisisbench_ml", "humset"):
        systems = res["results"][run]
        cells = [f"{systems[s]['ece']:.3f}" if s in systems else "—" for s in SYSTEMS]
        rows.append(f"| {RUN_NAMES[run]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(out: Path):
    res = json.loads(Path("results/t05_test.json").read_text())
    ft = json.loads(Path("results/t07_finetune.json").read_text())
    humaid_q = run_questions("humaid", "coarse_described")["coarse_described/label"]
    haiti_q = Q.haiti_variants()["noul_described"]["water_food"]
    urg = {
        k: round(v["urgency_auc"]["value"], 3) for k, v in res["results"]["haiti_original"].items()
    }
    card = f"""---
license: cc-by-nc-sa-4.0
base_model: convaiinnovations/laya
language: [en, fr, es, it, pt, tl, ht]
pipeline_tag: text-classification
tags: [laya, crisis-informatics, humanitarian, disaster-response, calibration, multilingual]
datasets:
  - QCRI/HumAID-events
  - QCRI/CrisisBench-all-lang
  - nlp-thedeep/humset
  - community-datasets/disaster_response_messages
---

# laya-crisis-triage

[Laya](https://github.com/NandhaKishorM/laya)'s multilingual checkpoint (mmBERT-base, 322M),
fine-tuned to triage disaster messages: what people need, which kind of information a crisis
tweet gives, and which humanitarian sector a report excerpt is about. One model answers all of
these typed questions in one forward pass, in any of the languages above, with calibrated
probabilities.

Research artefact from [crisis-triage]({REPO}), where every number below comes from. **It is
not an emergency system:** keep a human in the loop for any real decision about people in need.

## Use

```python
# pip install laya==0.3.5
import laya

agent = laya.load("mazenDDr/laya-crisis-triage")
questions = {{
    "kind": {json.dumps(humaid_q, indent=4).replace(chr(10), chr(10) + "    ")},
    "water_food": {json.dumps(haiti_q)},
}}
res = agent.predict({{"message": "We have had no drinking water for two days in Leogane"}}, questions)
print(res["answers"]["kind"]["probabilities"], res["answers"]["water_food"]["noul"])
```

It was trained on exactly these question wordings (and the other needs and sectors in
[`questions.py`]({REPO}/blob/main/src/crisis_triage/questions.py)). Other wordings still work,
since it is Laya, but were not measured.

## Results on held-out test data

Every system answers the same test messages. The splits are by time where there are several
disasters (tuned on 2016-17 / 2011-12 events, tested on later ones). Macro-F1 is over 8 shared
classes; mean AUC is over 4 needs (Haiti) or 10 sectors (HumSet). 95% intervals and paired
differences are in [`results/t05_test.json`]({REPO}/blob/main/results/t05_test.json).

{results_table(res)}

Calibration error (ECE, lower is better):

{ece_table(res)}

Baselines: e5 + LR is multilingual-e5-base embeddings with logistic regression, trained on the
same dev data. Qwen3-4B-Instruct-2507 and Gemma-3-4B-it are zero-shot, asked the same
questions and read out from next-token probabilities.

## Training

- Base: `convaiinnovations/laya`, `multilingual` subfolder.
- Data: dev splits only, {ft["train_examples"]:,} (message, question) examples: Haiti 2010 SMS
  needs (original or English text), HumAID and CrisisBench tweet classes, HumSet sectors. A fixed
  dev sample was held out for the temperature fit and the decision thresholds; test data was
  never used.
- Recipe: Laya's own (RLCD policy gradient with a proper-scoring reward, plus soft
  cross-entropy), {len(ft["epochs"])} epochs, {ft["epochs"][-1]["seconds"] / 60:.0f} minutes
  on one RTX 5060 Ti, bf16.
- Held-out loss: yes/no {ft["held_out_loss"]["before"]["noul"]} → {ft["held_out_loss"]["after_temperature"]["noul"]},
  choice {ft["held_out_loss"]["before"]["choice"]} → {ft["held_out_loss"]["after_temperature"]["choice"]}
  (before → after, with the fitted temperature).

## Limitations

- **Urgency was not trained and drifted:** on the Haiti SMS, urgency AUC is {urg["laya-ft"]}
  for this model vs {urg["laya"]} for zero-shot Laya (against a stand-in label built from rescue
  and medical needs).
- The Haiti SMS set is one event with a message-level split, and HumSet's test shares projects
  with its training data; both favour trained models. HumAID and CrisisBench are the
  new-disaster tests.
- Only 11 Haiti test messages ask for rescue, so that need's score is noisy.
- Haitian Creole is read far better through a translation (see the repository).
- Speed, one message at a time on an RTX 5060 Ti under WSL: 16-53 ms (the GPU is bimodal
  between runs).

## Licence

CC BY-NC-SA 4.0, because the model was trained on HumAID and CrisisBench (CC BY-NC-SA 4.0).
The base model Laya is Apache-2.0 and HumSet is Apache-2.0. The Disaster Response Messages
dataset (Appen) states no licence on its dataset card.
"""
    out.write_text(card)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "MODEL_CARD.md"))
