from crisis_triage.questions import TRIAGE


def test_every_question_has_a_known_type():
    assert {q["type"] for q in TRIAGE.values()} <= {"choice", "score", "noul"}


def test_need_has_a_none_option():
    assert "none" in TRIAGE["need"]["criteria"]
