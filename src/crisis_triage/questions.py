"""The typed triage questions Laya answers for every message.

This is the first draft used by the smoke test. The final wording is chosen on dev data
(T03), because question wording changed Laya's accuracy a lot in earlier work.
"""

from __future__ import annotations

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
