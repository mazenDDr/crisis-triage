from crisis_triage.latency import slug, summarize, time_call


def test_summarize_percentiles():
    s = summarize([float(i) for i in range(1, 101)])
    assert s["min_ms"] == 1.0
    assert 50.0 <= s["p50_ms"] <= 51.0
    assert 95.0 <= s["p95_ms"] <= 96.0


def test_time_call_counts_runs():
    calls = []
    t = time_call(lambda: calls.append(1), warmup=3, runs=7)
    assert len(calls) == 10
    assert t["runs"] == 7 and t["warmup"] == 3


def test_slug_drops_spaces_and_symbols():
    assert slug(" RTX 5060 Ti ") == "rtx5060ti"
