"""Step 3 - features built only from information available at the close of day t.

Target: the log return on the next trading day (t+1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .egarch import sigma_next

BASE_FEATURES = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
    "ret_mean_5", "ret_mean_20", "ret_std_5", "ret_std_20",
    "mom_20", "ma_gap_20", "rsi_14",
    "range_hl", "range_mean_5", "close_pos",
    "log_volume", "volume_ratio_20", "zero_volume", "zero_volume_share_20",
    "dist_12m_high", "dist_12m_low", "day_of_week",
]
VOL_FEATURES = ["egarch_sigma", "egarch_sigma_ratio"]


def stock_dummies(codes: list[str]) -> list[str]:
    return [f"stock_{c}" for c in sorted(codes)]


def _rsi(r: pd.Series, n: int = 14) -> pd.Series:
    up = r.clip(lower=0).rolling(n, min_periods=n).mean()
    down = (-r.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def build_stock_features(g: pd.DataFrame, egarch_fit: dict, cfg: dict) -> pd.DataFrame:
    """g = output of clean.prepare_series for one stock."""
    g = g.copy()
    r = g["ret"]
    rz = r.fillna(0.0)
    for k in range(1, 6):
        g[f"ret_lag{k}"] = r.shift(k - 1)
    g["ret_mean_5"] = rz.rolling(5, min_periods=3).mean()
    g["ret_mean_20"] = rz.rolling(20, min_periods=10).mean()
    g["ret_std_5"] = rz.rolling(5, min_periods=3).std()
    g["ret_std_20"] = rz.rolling(20, min_periods=10).std()
    lp = np.log(g["adj_price"])
    g["mom_20"] = lp - lp.shift(20)
    g["ma_gap_20"] = lp - np.log(g["adj_price"].rolling(20, min_periods=10).mean())
    g["rsi_14"] = _rsi(rz)
    g["range_hl"] = (g["day_high"] - g["day_low"]) / g["price"]
    g["range_mean_5"] = g["range_hl"].rolling(5, min_periods=3).mean()
    span = (g["day_high"] - g["day_low"]).replace(0, np.nan)
    g["close_pos"] = ((g["price"] - g["day_low"]) / span).fillna(0.5)
    g["log_volume"] = np.log1p(g["volume"])
    g["volume_ratio_20"] = g["log_volume"] - g["log_volume"].rolling(20, min_periods=10).mean()
    g["zero_volume"] = (g["volume"] == 0).astype(float)
    g["zero_volume_share_20"] = g["zero_volume"].rolling(20, min_periods=10).mean()
    g["dist_12m_high"] = np.log(g["price"] / g["high_12m"])
    g["dist_12m_low"] = np.log(g["price"] / g["low_12m"])
    g["day_of_week"] = g["date"].dt.day_of_week.astype(float)

    g["egarch_sigma"] = sigma_next(r, egarch_fit, cfg)
    g["egarch_sigma_ratio"] = g["egarch_sigma"] / g["ret_std_20"].replace(0, np.nan)

    # Targets / evaluation columns (future values - never used as inputs).
    g["target_ret"] = r.shift(-1)
    g["target_date"] = g["date"].shift(-1)
    g["next_price"] = g["price"].shift(-1)
    return g


def build_features(clean_df: pd.DataFrame, egarch_fits: dict, cfg: dict) -> pd.DataFrame:
    parts = [build_stock_features(g, egarch_fits[code], cfg)
             for code, g in clean_df.groupby("code", sort=True)]
    df = pd.concat(parts, ignore_index=True)
    for c in cfg["stocks"]:
        df[f"stock_{c}"] = (df["code"] == c).astype(float)
    return df
