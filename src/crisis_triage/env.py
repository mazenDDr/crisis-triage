"""Record what each machine can run: library versions and the GPU."""

from __future__ import annotations

import importlib
import platform

from crisis_triage.latency import machine_label

LIBRARIES = ("numpy", "torch", "transformers", "laya", "datasets")


def library_versions(names: tuple[str, ...] = LIBRARIES) -> dict[str, str | None]:
    """Version of each library, or None when it is not installed."""
    versions = {}
    for name in names:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


def torch_devices() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"cuda": False, "mps": False}
    info: dict[str, object] = {"cuda": torch.cuda.is_available(), "mps": False}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["device"] = props.name
        info["capability"] = f"sm_{props.major}{props.minor}"
        info["vram_gib"] = round(props.total_memory / 2**30, 1)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        info["mps"] = True
    return info


def collect() -> dict[str, object]:
    return {
        "machine": machine_label(),
        "system": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "libraries": library_versions(),
        "torch": torch_devices(),
    }
