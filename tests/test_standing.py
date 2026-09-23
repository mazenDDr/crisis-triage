from crisis_triage.standing import standing


def _results(ft, e5, zs=0.5, ece_ft=0.01, ece_e5=0.05):
    mk = lambda f1, ece: {"macro_f1": {"value": f1}, "ece": ece}  # noqa: E731
    return {"humaid": {"laya-ft": mk(ft, ece_ft), "e5lr": mk(e5, ece_e5), "laya": mk(zs, 0.2)}}


def _paired(lo, hi):
    return {"humaid": {"laya-ft - e5lr": {"value": (lo + hi) / 2, "lo": lo, "hi": hi}}}


def test_best_only_when_the_interval_is_above_zero():
    assert standing(_results(0.73, 0.70), _paired(0.01, 0.05))["ft_standing"] == "the best of five"
    tie = standing(_results(0.73, 0.72), _paired(-0.01, 0.02))["ft_standing"]
    assert tie.startswith("tied for the best")
    assert standing(_results(0.70, 0.73), _paired(-0.05, -0.01))["ft_standing"].startswith("second")


def test_trust_claim_follows_the_lowest_calibration_error():
    assert (
        "is the lowest (0.010; next: a tuned e5 classifier, 0.050)"
        in standing(_results(0.73, 0.70), _paired(0.01, 0.05))["ft_trust"]
    )
    worse = standing(_results(0.73, 0.70, ece_ft=0.08, ece_e5=0.02), _paired(0.01, 0.05))
    assert "lowest: a tuned e5 classifier" in worse["ft_trust"]
    assert (
        standing(_results(0.73, 0.70), _paired(0.01, 0.05))["zs_standing"] == "the weakest of five"
    )
