from crisis_triage.questions import (
    HUMANITARIAN,
    URGENCY,
    class_variants,
    flatten,
    haiti_variants,
    humset_variants,
)


def test_every_question_has_a_known_type():
    flat = flatten(haiti_variants(), URGENCY) | flatten(humset_variants())
    assert {q["type"] for q in flat.values()} <= {"choice", "score", "noul"}


def test_flat_keys_are_variant_slash_question():
    flat = flatten(haiti_variants(), URGENCY)
    assert "noul_short/rescue" in flat and "urgency_plain/score" in flat
    assert all(k.count("/") == 1 for k in flat)


def test_class_variants_cover_every_label():
    labels = ["requests_or_urgent_needs", "not_humanitarian"]
    v = class_variants(labels)
    assert v["choice_names"]["label"]["criteria"] == [
        "requests or urgent needs",
        "not humanitarian",
    ]
    assert set(v["choice_described"]["label"]["criteria"].values()) == {
        HUMANITARIAN[c] for c in labels
    }


def test_every_fine_class_maps_to_a_coarse_class():
    from crisis_triage.questions import COARSE, TO_COARSE

    assert set(TO_COARSE) == set(HUMANITARIAN)
    assert set(TO_COARSE.values()) == set(COARSE)
