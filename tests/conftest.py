"""Shared fixtures: synthetic data + a fast config, trained once per test session."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def workdir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("run")
    subprocess.run([sys.executable, str(ROOT / "scripts/make_sample_data.py"),
                    "--out", str(d / "raw"), "--start", "2021", "--end", "2026"], check=True)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["paths"] = {"raw_dir": str(d / "raw"), "processed_dir": str(d / "processed"),
                    "models_dir": str(d / "models"), "reports_dir": str(d / "reports")}
    cfg["split"]["train_end"] = "2024-06-30"
    cfg["xgboost"]["grid"] = {"max_depth": [2], "learning_rate": [0.05],
                              "n_estimators": [50], "min_child_weight": [20]}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    return d


@pytest.fixture(scope="session")
def trained(workdir) -> Path:
    from nse.pipeline import main
    main(["--config", str(workdir / "config.yaml")])
    return workdir
