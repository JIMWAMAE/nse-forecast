"""Leakage audit: features for day t must not change when future rows are removed."""
import numpy as np
import pandas as pd

from nse.clean import clean
from nse.config import load_config
from nse.features import BASE_FEATURES, VOL_FEATURES, build_features
from nse.ingest import ingest
from nse.pipeline import fit_all_egarch


def test_features_do_not_use_future(workdir):
    cfg = load_config(workdir / "config.yaml")
    data, _ = clean(ingest(cfg["paths"]["raw_dir"]), cfg)
    fits = fit_all_egarch(data, cfg["split"]["train_end"], cfg)
    cutoff = pd.Timestamp("2025-06-30")
    full = build_features(data, fits, cfg)
    cut = build_features(data[data["date"] <= cutoff], fits, cfg)
    cols = BASE_FEATURES + VOL_FEATURES
    a = full[full["date"] <= cutoff].set_index(["code", "date"])[cols].sort_index()
    b = cut.set_index(["code", "date"])[cols].sort_index()
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-6, atol=1e-9)


def test_target_is_next_day(workdir):
    cfg = load_config(workdir / "config.yaml")
    data, _ = clean(ingest(cfg["paths"]["raw_dir"]), cfg)
    fits = fit_all_egarch(data, cfg["split"]["train_end"], cfg)
    f = build_features(data, fits, cfg)
    g = f[f["code"] == "SCOM"].reset_index(drop=True)
    assert (g["target_date"].iloc[:-1].to_numpy() == g["date"].iloc[1:].to_numpy()).all()
    assert np.allclose(g["target_ret"].iloc[:-1].fillna(0), g["ret"].iloc[1:].fillna(0))
