"""Seeds and run provenance (software versions, git hash, config hash, hardware)."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_PACKAGES = [
    "numpy",
    "pandas",
    "torch",
    "scikit-learn",
    "xgboost",
    "statsmodels",
    "scipy",
    "yfinance",
    "pandas_market_calendars",
]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - global seed for third-party code
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_hash(repo: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def environment_info() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for p in _PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = "not installed"
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": versions,
        "colab": "COLAB_RELEASE_TAG" in os.environ,
    }
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cudnn"] = torch.backends.cudnn.version()
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    except ImportError:
        pass
    return info


def write_run_manifest(run_dir: Path, cfg: dict[str, Any], seed: int | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git": git_hash(),
        "config_hash": config_hash(cfg),
        "seed": seed,
        "config": cfg,
        "environment": environment_info(),
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
