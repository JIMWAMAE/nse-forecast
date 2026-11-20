"""Load config.yaml and resolve paths relative to the project root."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("NSE_CONFIG", ROOT / "config.yaml"))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = path.resolve().parent
    for key, val in cfg["paths"].items():
        p = Path(val)
        cfg["paths"][key] = p if p.is_absolute() else base / p
    return cfg
