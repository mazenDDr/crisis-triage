import numpy as np

from crisis_triage.llm import prompt_parts, to_answer
from crisis_triage.questions import URGENCY, choice_question, haiti_variants


def test_choice_prompt_letters_and_keys():
    q = choice_question(["not_humanitarian", "requests_or_needs"], "described")
    user, tokens, keys = prompt_parts(q, "we need water")
    assert tokens == ["A", "B"]
    assert keys == ["not humanitarian", "requests or needs"]
    assert "A) not humanitarian: not about the disaster" in user
    assert '"""we need water"""' in user


def test_noul_and_score_answers():
    noul = haiti_variants()["noul_described"]["rescue"]
    _, tokens, keys = prompt_parts(noul, "help")
    assert tokens == ["Yes", "No"]
    assert to_answer(noul, np.array([0.8, 0.2]), keys) == {"p": 0.8}
    score = URGENCY["urgency_concrete"]
    _, tokens, keys = prompt_parts(score, "help")
    ans = to_answer(score, np.array([0.0, 0.5, 0.5]), keys)
    assert tokens == ["0", "1", "2"] and abs(ans["score"] - 1.5) < 1e-9


def test_long_messages_are_cut():
    q = haiti_variants()["noul_short"]["medical"]
    user, _, _ = prompt_parts(q, "x" * 5000)
    assert user.count("x") == 2000
