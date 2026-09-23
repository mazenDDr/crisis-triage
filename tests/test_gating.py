import numpy as np

from crisis_triage.gating import (
    aurc,
    choice_decisions,
    gate,
    multilabel_decisions,
    risk_coverage,
    threshold_for,
)


def test_choice_decisions():
    conf, ok = choice_decisions(np.array([[0.7, 0.3], [0.4, 0.6]]), np.array([0, 0]))
    assert conf.tolist() == [0.7, 0.6] and ok.tolist() == [True, False]


def test_multilabel_needs_every_label_right():
    p = np.array([[0.9, 0.1], [0.9, 0.8]])
    y = np.array([[1, 0], [1, 0]])
    conf, ok = multilabel_decisions(p, y, np.array([0.5, 0.5]))
    assert np.allclose(conf, [0.9, 0.8]) and ok.tolist() == [True, False]


def test_threshold_reaches_target_with_most_coverage():
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.6])
    ok = np.array([True, True, True, False, True])
    cut = threshold_for(conf, ok, 0.95)
    assert cut == 0.9
    assert gate(conf, ok, cut) == (0.6, 1.0)
    assert threshold_for(conf, np.zeros(5, bool), 0.95) == float("inf")


def test_risk_coverage_and_aurc():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    ok = np.array([True, True, False, False])
    curve = risk_coverage(conf, ok, points=4)
    assert [c["accuracy"] for c in curve] == [1.0, 1.0, 2 / 3, 0.5]
    assert abs(aurc(conf, ok) - np.mean([0, 0, 1 / 3, 1 / 2])) < 1e-9


def test_threshold_keeps_or_drops_tied_confidences_together():
    # three answers tied at 1.0, only two right: the tie group is 67% accurate as a whole
    conf = np.array([1.0, 1.0, 1.0, 0.9])
    ok = np.array([True, True, False, True])
    assert threshold_for(conf, ok, 0.95) == float("inf")
    assert threshold_for(conf, ok, 0.6) == 0.9
    assert gate(conf, ok, threshold_for(conf, ok, 0.6)) == (1.0, 0.75)
