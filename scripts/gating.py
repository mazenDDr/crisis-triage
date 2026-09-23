"""T06: how many messages can each system route without a person, at 90% and 95% accuracy?

For every run and system: the confidence cut that reaches the target on the dev sample, applied
to the test messages every system answered (coverage and accuracy reached, 95% intervals); the
naive rule "trust the model's own confidence >= target"; the risk-coverage curve and its area.
Results: results/t06_gating.json.
"""

import json
from pathlib import Path

import numpy as np

from crisis_triage import gating as G
from crisis_triage import questions as Q
from crisis_triage.laya_run import load
from crisis_triage.metrics import best_threshold, paired_bootstrap
from crisis_triage.runs import RUNS, SETTINGS, test_split
from crisis_triage.scoring import dev_sample, multilabel_scores

TARGETS = (0.90, 0.95)
SYSTEMS = ("laya", "laya-ft", "e5lr", "qwen3-4b", "gemma3-4b")
# Zero-shot Laya's dev answers with the chosen settings (T03 / T03b runs).
LAYA_DEV = {
    "haiti_original": "outputs/t03/haiti_sms__orig_multilingual.jsonl",
    "haiti_mt": "outputs/t03b/laya_nllb600m_lower.jsonl",
    "haiti_human": "outputs/t03/haiti_sms__en_typed.jsonl",
    "humaid": "outputs/t03/humaid__typed.jsonl",
    "crisisbench_ml": "outputs/t03/crisisbench_ml__router.jsonl",
    "humset": "outputs/t03/humset__router.jsonl",
}


def paths(system: str, name: str) -> tuple[Path, Path]:
    if system == "laya":
        return Path(LAYA_DEV[name]), Path("outputs/t04") / f"{name}.jsonl"
    base = Path("outputs/t07") if system == "laya-ft" else Path("outputs/t05") / system
    return base / f"{name}__dev.jsonl", base / f"{name}__test.jsonl"


def decisions(track, variant, df, recs, thresholds=None):
    """(confidence, correct, thresholds) for the rows of df, from one system's answers."""
    answers = [recs[u]["answers"] for u in df["uid"]]
    if track in ("haiti_sms", "humset"):
        labels = list(Q.HAITI_NEEDS if track == "haiti_sms" else Q.HUMSET_SECTORS)
        y = np.array([[c in set(ls) for c in labels] for ls in df["labels"]])
        p = multilabel_scores(answers, variant, labels)
        if thresholds is None:  # best-F1 cut per label, chosen on dev
            thresholds = np.array([best_threshold(y[:, j], p[:, j]) for j in range(len(labels))])
        conf, ok = G.multilabel_decisions(p, y, thresholds)
        return conf, ok, thresholds
    classes = [Q.readable(c) for c in Q.COARSE]
    probs = np.array([[a[f"{variant}/label"]["probs"][c] for c in classes] for a in answers])
    gold = np.array([classes.index(Q.readable(Q.TO_COARSE[g])) for g in df["label"]])
    conf, ok = G.choice_decisions(probs, gold)
    return conf, ok, None


def main():
    out = {"targets": list(TARGETS), "runs": {}}
    for name, (track, setting, _) in RUNS.items():
        variant = SETTINGS[setting]["variant"]
        systems = [s for s in SYSTEMS if all(p.exists() for p in paths(s, name))]
        tests = {s: load(paths(s, name)[1]) for s in systems}
        common = set.intersection(*(set(t) for t in tests.values()))
        te = test_split(track)
        te = te[te["uid"].isin(common)].reset_index(drop=True)
        dev = dev_sample(track)
        run_out = {"n_test": len(te), "n_dev": len(dev), "systems": {}, "paired_coverage": {}}
        kept = {}  # (system, target) -> which test messages are sent automatically
        for s in systems:
            dconf, dok, thr = decisions(track, variant, dev, load(paths(s, name)[0]))
            conf, ok, _ = decisions(track, variant, te, tests[s], thr)
            r = {
                "accuracy_all": float(ok.mean()),
                "aurc": G.aurc(conf, ok),
                "curve": G.risk_coverage(conf, ok),
                "at": {},
            }
            for target in TARGETS:
                cut = G.threshold_for(dconf, dok, target)
                cov, acc = G.gate(conf, ok, cut)
                kept[(s, target)] = conf >= cut
                ncov, nacc = G.gate(conf, ok, target)  # trust the model's own confidence
                r["at"][str(target)] = {
                    "dev_cut": cut,
                    "coverage": paired_bootstrap(
                        lambda i, c=cut, cf=conf: float((cf[i] >= c).mean()), len(te)
                    ),
                    "accuracy_covered": acc,
                    "accuracy_covered_ci": paired_bootstrap(
                        lambda i, c=cut, cf=conf, o=ok: (
                            float(o[i][cf[i] >= c].mean()) if (cf[i] >= c).any() else float("nan")
                        ),
                        len(te),
                    ),
                    "coverage_point": cov,
                    "naive": {"coverage": ncov, "accuracy_covered": nacc},
                }
            run_out["systems"][s] = r
            print(
                name, s, {t: round(v["coverage_point"], 3) for t, v in r["at"].items()}, flush=True
            )
        if "laya-ft" in systems:  # fine-tuned Laya's coverage minus each other system's
            for target in TARGETS:
                a = kept[("laya-ft", target)]
                run_out["paired_coverage"][str(target)] = {
                    f"laya-ft - {s}": paired_bootstrap(
                        lambda i, a=a, b=kept[(s, target)]: float(a[i].mean() - b[i].mean()),
                        len(te),
                    )
                    for s in systems
                    if s != "laya-ft"
                }
        out["runs"][name] = run_out
    Path("results/t06_gating.json").write_text(json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
