import numpy as np

from crisis_triage.metrics import auc, binary_f1, macro_f1, paired_bootstrap


def test_auc_perfect_random_and_ties():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auc([0, 1], [0.5, 0.5]) == 0.5
    assert np.isnan(auc([1, 1], [0.1, 0.2]))


def test_macro_f1_averages_classes():
    # class a: tp 1, fp 1 -> 2/3; class b: tp 1, fn 1 -> 2/3
    assert abs(macro_f1(["a", "b", "b"], ["a", "b", "a"]) - 2 / 3) < 1e-9
    assert macro_f1(["a", "b"], ["a", "a"], labels=["a", "b", "c"]) == (2 / 3 + 0 + 0) / 3


def test_binary_f1_threshold():
    assert binary_f1([1, 0, 1], [0.9, 0.6, 0.4]) == 0.5


def test_paired_bootstrap_interval_contains_value():
    x = np.random.default_rng(1).normal(0.1, 1, 500)
    r = paired_bootstrap(lambda i: float(x[i].mean()), len(x), reps=300)
    assert r["lo"] <= r["value"] <= r["hi"]


def test_ece_zero_when_confidence_matches_accuracy():
    from crisis_triage.metrics import ece

    conf = [0.8] * 10
    correct = [1] * 8 + [0] * 2
    assert abs(ece(conf, correct)) < 1e-9
    assert abs(ece([0.9] * 10, correct) - 0.1) < 1e-9


def test_brier_multiclass_and_binary():
    from crisis_triage.metrics import brier

    assert brier([[1, 0], [0.5, 0.5]], [[1, 0], [1, 0]]) == 0.25
    assert brier([0.5, 1.0], [1, 1]) == 0.125


def test_best_threshold_separates_classes():
    from crisis_triage.metrics import best_threshold, binary_f1

    y = [0] * 50 + [1] * 50
    s = list(np.linspace(0, 0.4, 50)) + list(np.linspace(0.6, 1, 50))
    assert binary_f1(y, s, best_threshold(y, s)) == 1.0
