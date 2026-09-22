"""One timing protocol: warm-up, one request at a time, then p50/p95 over many runs.

A latency number is only reported together with the machine label, the model and the run
counts, so every record carries all of them.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import time
from collections.abc import Callable

import numpy as np


def machine_label() -> str:
    """Short hardware label such as `mac-m4pro` or `rtx5060ti`. Never the hostname."""
    if label := os.environ.get("CRISIS_MACHINE"):
        return label
    try:
        import torch

        if torch.cuda.is_available():
            return slug(torch.cuda.get_device_name(0).replace("NVIDIA", "").replace("GeForce", ""))
    except ImportError:
        pass
    if platform.system() == "Darwin":
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout
        return "mac-" + slug(chip.replace("Apple", ""))
    return slug(platform.machine())


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def summarize(times_ms: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(float(np.percentile(times_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 3),
        "mean_ms": round(float(np.mean(times_ms)), 3),
        "min_ms": round(float(np.min(times_ms)), 3),
    }


def time_call(
    fn: Callable[[], object],
    warmup: int = 10,
    runs: int = 100,
    sync: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Time `fn` end to end. `sync` waits for the GPU (e.g. torch.cuda.synchronize)."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return {"machine": machine_label(), "warmup": warmup, "runs": runs, **summarize(times)}
