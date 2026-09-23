import numpy as np
import pandas as pd

from crisis_triage.finetune import examples, target
from crisis_triage.questions import COARSE, choice_question


def test_noul_and_choice_targets():
    assert target({"type": "noul"}, True) == [0.0, 1.0]
    assert target({"type": "noul"}, False) == [1.0, 0.0]
    q = choice_question(list(COARSE), "described", COARSE)
    t = target(q, "people affected")
    assert sum(t) == 1.0 and t[list(q["criteria"]).index("people affected")] == 1.0


def test_haiti_examples_one_per_need_without_urgency():
    df = pd.DataFrame(
        {"text": ["mwen bezwen dlo"], "text_en": ["I need water"], "labels": [["water_food"]]}
    )
    ex = examples("haiti_sms", df, np.random.default_rng(0))
    assert len(ex) == 4  # four needs, no urgency question
    hits = [e for e in ex if e["target"] == [0.0, 1.0]]
    assert len(hits) == 1 and "water" in hits[0]["q"]["instructions"]


def test_tweet_examples_use_the_coarse_class():
    df = pd.DataFrame({"text": ["3 dead"], "label": ["injured_or_dead_people"]})
    (ex,) = examples("humaid", df, np.random.default_rng(0))
    keys = list(ex["q"]["criteria"])
    assert keys[ex["target"].index(1.0)] == "people affected"
