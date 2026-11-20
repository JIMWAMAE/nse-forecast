"""XGBoost return models and chronological tuning."""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

log = logging.getLogger(__name__)


def make_model(params: dict, cfg: dict) -> xgb.XGBRegressor:
    p = {**cfg["xgboost"]["fixed"], **params}
    return xgb.XGBRegressor(random_state=cfg["seed"], n_jobs=-1, **p)


def grid(cfg: dict) -> list[dict]:
    g = cfg["xgboost"]["grid"]
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def tune(train: pd.DataFrame, val: pd.DataFrame, features: list[str], cfg: dict,
         label: str) -> tuple[dict, pd.DataFrame]:
    """Try each grid point: fit on train, score on validation (RMSE of next-day return)."""
    rows = []
    for params in grid(cfg):
        m = make_model(params, cfg).fit(train[features], train["target_ret"])
        pred = m.predict(val[features])
        hit = np.sign(pred) == np.sign(val["target_ret"])
        nz = val["target_ret"] != 0
        rows.append({**params, "model": label,
                     "val_rmse": rmse(val["target_ret"], pred),
                     "val_dir_acc": float(hit[nz].mean())})
    res = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    best = {k: res.iloc[0][k] for k in cfg["xgboost"]["grid"]}
    best = {k: (int(v) if float(v).is_integer() and k != "learning_rate" else float(v))
            for k, v in best.items()}
    log.info("%s best params %s  val RMSE %.5f", label, best, res.iloc[0]["val_rmse"])
    return best, res
