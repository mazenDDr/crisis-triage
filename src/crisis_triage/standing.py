"""Where Laya stands among the five systems, worded from the results, never by hand.

The README and the page both use these phrases, so a claim like "the best of five" only
appears when the paired interval on the same test messages supports it.
"""

from __future__ import annotations

LABELS = {
    "laya-ft": "fine-tuned Laya",
    "laya": "Laya out of the box",
    "e5lr": "a tuned e5 classifier",
    "qwen3-4b": "Qwen3-4B",
    "gemma3-4b": "Gemma-3-4B",
}


def standing(results: dict, paired: dict, run: str = "humaid") -> dict:
    """Phrases for one run, from t05_test.json's `results` and `paired` sections."""
    res = results[run]
    f1 = {k: v["macro_f1"]["value"] for k, v in res.items()}
    ece = {k: v["ece"] for k, v in res.items()}
    others = [k for k in f1 if k != "laya-ft"]
    rival = max(others, key=f1.get)
    diff = paired[run][f"laya-ft - {rival}"]
    if diff["lo"] > 0:
        ft = "the best of five"
    elif diff["hi"] >= 0:
        ft = f"tied for the best with {LABELS[rival]}"
    else:
        ft = f"second to {LABELS[rival]}"
    order = sorted(f1, key=f1.get)
    zs_place = order.index("laya") + 1
    zs = "the weakest of five" if zs_place == 1 else f"number {len(order) - zs_place + 1} of five"
    calm = min(ece, key=ece.get)
    runner = min((k for k in ece if k != calm), key=ece.get)
    if calm == "laya-ft":
        trust = (
            f"its calibration error is the lowest ({ece['laya-ft']:.3f}; "
            f"next: {LABELS[runner]}, {ece[runner]:.3f})"
        )
    else:
        trust = (
            f"its calibration error is {ece['laya-ft']:.3f} "
            f"(lowest: {LABELS[calm]}, {ece[calm]:.3f})"
        )
    return {
        "ft_standing": ft,
        "ft_rival": LABELS[rival],
        "ft_vs_rival": f"{diff['value']:+.3f} [{diff['lo']:+.3f}, {diff['hi']:+.3f}]",
        "zs_standing": zs,
        "ft_trust": trust,
    }
