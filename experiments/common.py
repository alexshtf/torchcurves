"""Shared recording and environment setup for the paper experiments."""

# ruff: noqa: E402

import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLCONFIGDIR"] = str(Path(__file__).parent / ".cache" / "matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import csv
import json
import platform
import subprocess
from importlib.metadata import version

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
BASELINE_COMMIT = "7b6ce1c87f18c8bc90c208f6b494042344216b11"


def prepare() -> None:
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def metadata() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    cpuinfo = Path("/proc/cpuinfo")
    cpu = platform.processor()
    if not cpu and cpuinfo.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )

    return {
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("torchcurves", "torch", "numpy", "scipy", "matplotlib", "efficient-kan")
        },
        "torchcurves_commit": git("rev-parse", "HEAD"),
        "torchcurves_dirty": bool(git("status", "--porcelain")),
        "efficient_kan_commit": BASELINE_COMMIT,
        "cpu": cpu or "unknown",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda,
        "threads": {
            name: os.environ[name]
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "torch_threads": torch.get_num_threads(),
        "seed": SEED,
    }


def summary(values: list[float]) -> dict:
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"median_ms": float(median), "q1_ms": float(q1), "q3_ms": float(q3)}


def save(directory: Path, name: str, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (directory / f"{name}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
