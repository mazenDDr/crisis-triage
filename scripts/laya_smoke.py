"""Smoke test: Laya answers the triage questions on a few hand-written messages, then is timed.

The messages are made up to check the plumbing, languages and speed. They are not results:
accuracy is only measured on the labelled datasets (T04).
"""

import argparse
import json
from pathlib import Path

from crisis_triage.latency import time_call

# First draft of the triage questions, kept as the smoke test's input. T03 chooses the wording.
TRIAGE = {
    "need": {
        "type": "choice",
        "instructions": "What does the person who wrote this message need most?",
        "criteria": {
            "rescue": "trapped, missing people, search and rescue",
            "medical": "injured, sick, medicine, doctors",
            "water_food": "drinking water, food, hunger",
            "shelter": "tents, housing, homeless after the disaster",
            "none": "no request for help, news, opinion or unrelated",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this message for emergency responders?",
        "criteria": ["not urgent", "needs help soon", "life in danger right now"],
    },
    "has_location": {
        "type": "noul",
        "instructions": "Does the message say where the people who need help are?",
    },
}

MESSAGES = {
    "en": "We are trapped under the school roof in Carrefour, 3 children, please send help",
    "fr": "Nous n'avons plus d'eau potable depuis deux jours à Léogâne",
    "ht": "Nou bezwen tant ak manje, kay nou kraze nan Pòtoprens",
    "es": "Mi madre está herida y necesita un médico urgente",
    "ar": "الماء يغمر البيت ونحن محاصرون في الطابق الثاني، حي المعادي",
    "none": "Thoughts and prayers to everyone affected by the earthquake tonight",
}

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda")
parser.add_argument("--runs", type=int, default=100)
args = parser.parse_args()

import torch  # noqa: E402
from laya import Router  # noqa: E402

router = Router(preload=True, device=args.device)
rows = []
for lang, text in MESSAGES.items():
    res = router.predict({"message": text}, TRIAGE)
    a = res["answers"]
    rows.append(
        {
            "lang": lang,
            "model": res["routing"]["model"],
            "need": a["need"]["choice"],
            "need_conf": round(a["need"]["confidence"], 3),
            "urgency": round(a["urgency"]["score"], 2),
            "has_location": round(a["has_location"]["noul"], 3),
        }
    )
    print(json.dumps(rows[-1], ensure_ascii=False))

sync = torch.cuda.synchronize if args.device == "cuda" else None
timing = {}
for lang in ("en", "ar"):
    state = {"message": MESSAGES[lang]}
    t = time_call(lambda s=state: router.predict(s, TRIAGE), runs=args.runs, sync=sync)
    timing[lang] = t
    print(lang, json.dumps(t))

out = Path("outputs") / f"laya_smoke_{timing['en']['machine']}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"answers": rows, "timing": timing}, indent=2, ensure_ascii=False))
print(f"saved {out}")
