from __future__ import annotations

from pathlib import Path
import json
import os
import random
import time

import numpy as np
import torch


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dirs() -> tuple[Path, Path]:
    root = project_root()
    results_dir = root / "results"
    logs_dir = root / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, logs_dir


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Safe guard for newer PyTorch versions if available
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def now_ts() -> float:
    return time.time()


def minutes_elapsed(start_ts: float) -> float:
    return (time.time() - start_ts) / 60.0