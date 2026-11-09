"""Shared recording and settings for the paper experiments."""

import csv
import json
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
BASELINE_COMMIT = "7b6ce1c87f18c8bc90c208f6b494042344216b11"


def progress(total: int, study: str):
    return tqdm(
        total=total,
        desc=study,
        bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| {elapsed}{postfix}",
        dynamic_ncols=True,
        disable=None,
    )


def prepare() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def metadata() -> dict:
    return {
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("torchcurves", "torch", "numpy", "scipy", "matplotlib", "efficient-kan")
        },
        "torchcurves_commit": git("rev-parse", "HEAD"),
        "torchcurves_dirty": bool(git("status", "--porcelain")),
        "efficient_kan_commit": BASELINE_COMMIT,
        "cpu": cpu_model(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "torch_threads": torch.get_num_threads(),
        "seed": SEED,
    }


def summary(values: list[float]) -> dict:
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"median_ms": float(median), "q1_ms": float(q1), "q3_ms": float(q3)}


def write_rows(directory: Path, name: str, rows: list[dict]) -> None:
    (directory / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (directory / f"{name}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save(directory: Path, name: str, measurements: list[tuple[dict, list[float]]]) -> None:
    """Save each repetition and its summary in the same JSON and CSV formats."""
    raw, aggregates = [], []
    for config, times in measurements:
        raw.extend({**config, "repeat": repeat, "ms": value} for repeat, value in enumerate(times))
        aggregates.append({**config, **summary(times)})
    directory.mkdir(parents=True, exist_ok=True)
    write_rows(directory, f"{name}_raw", raw)
    write_rows(directory, f"{name}_summary", aggregates)
