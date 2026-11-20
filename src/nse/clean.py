"""Step 2 - keep the ten stocks, fix codes, remove bad rows and build clean returns.

Return definition
-----------------
ret_t = log(DayPrice_t / Previous_t)

'Previous' is the reference price the exchange used for that session. When the NSE
adjusts a price for a bonus issue or split, 'Previous' already carries the adjustment,
so using it keeps those jumps out of the return series. If 'Previous' is missing we
fall back to the last recorded close. Moves larger than `max_abs_return` are treated
as data errors and set to missing.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

NAME_KEYWORDS = {
    "ABSA": ["absa", "barclays"],
    "BRIT": ["britam"],
    "COOP": ["co-operative", "co-op", "cooperative"],
    "EQTY": ["equity group"],
    "KCB": ["kcb"],
    "KQ": ["kenya airways"],
    "KEGN": ["kengen", "electricity generating"],
    "KPLC": ["kenya power"],
    "KNRE": ["kenya re-insurance", "kenya reinsurance", "kenya-re"],
    "SCOM": ["safaricom"],
}


def map_codes(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    stocks = cfg["stocks"]
    aliases = cfg.get("code_aliases", {}) or {}
    code = df["code"].replace(aliases)
    unknown = ~code.isin(stocks)
    if "name" in df.columns and unknown.any():
        names = df.loc[unknown, "name"].str.lower()
        is_pref = names.str.contains("pref|%", regex=True)
        for target, words in NAME_KEYWORDS.items():
            hit = names.apply(lambda n: any(w in n for w in words)) & ~is_pref
            code.loc[hit[hit].index] = target
    out = df.assign(code=code)
    out = out[out["code"].isin(stocks)].copy()
    out["name"] = out["code"].map(stocks)
    return out


def prepare_series(g: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Clean one stock's rows (sorted by date) and add return columns.
    Used by the training pipeline AND the API, so both see identical logic."""
    c = cfg["cleaning"]
    g = g.sort_values("date").reset_index(drop=True).copy()
    for col in ["day_low", "day_high"]:
        if col not in g or g[col].isna().all():
            g[col] = g["price"]
        g[col] = g[col].where(g[col] > 0, g["price"])
    g["day_low"] = np.minimum(g["day_low"], g["price"])
    g["day_high"] = np.maximum(g["day_high"], g["price"])
    g["volume"] = g.get("volume", pd.Series(0.0, index=g.index)).fillna(0.0).clip(lower=0)
    for col in ["high_12m", "low_12m"]:
        if col not in g or g[col].isna().all():
            g[col] = np.nan
    # Fallback 12-month range from history where the file lacks it.
    roll_hi = g["day_high"].rolling(250, min_periods=1).max()
    roll_lo = g["day_low"].rolling(250, min_periods=1).min()
    g["high_12m"] = g["high_12m"].where(g["high_12m"] > 0, roll_hi)
    g["low_12m"] = g["low_12m"].where(g["low_12m"] > 0, roll_lo)

    last_close = g["price"].shift(1)
    prev = g["previous"] if "previous" in g else pd.Series(np.nan, index=g.index)
    ref = prev.where(prev > 0, last_close)
    g["ret"] = np.log(g["price"] / ref)
    g["adjusted"] = (np.abs(np.log(prev / last_close)) > c["adjustment_tolerance"]).fillna(False)
    g["outlier"] = g["ret"].abs() > c["max_abs_return"]
    g.loc[g["outlier"], "ret"] = np.nan

    # Adjusted price index (continuous through corporate actions), anchored to the latest close.
    cum = g["ret"].fillna(0.0).cumsum()
    g["adj_price"] = g["price"].iloc[-1] * np.exp(cum - cum.iloc[-1])
    return g


def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    n0 = len(df)
    df = map_codes(df, cfg)
    df = df[df["date"].notna()]
    bad_price = df["price"].isna() | (df["price"] <= 0)
    dropped_price = df[bad_price].groupby("code").size()
    df = df[~bad_price]
    dup = df.duplicated(["code", "date"], keep="last")
    dropped_dup = df[dup].groupby("code").size()
    df = df[~dup]

    parts = [prepare_series(g, cfg) for _, g in df.groupby("code", sort=True)]
    out = pd.concat(parts, ignore_index=True)

    q = out.groupby("code").agg(
        name=("name", "first"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        rows=("date", "size"),
        zero_volume_days=("volume", lambda v: int((v == 0).sum())),
        price_adjustments=("adjusted", "sum"),
        outlier_returns=("outlier", "sum"),
    )
    q["dropped_missing_price"] = dropped_price.reindex(q.index).fillna(0).astype(int)
    q["dropped_duplicates"] = dropped_dup.reindex(q.index).fillna(0).astype(int)
    q["first_date"] = q["first_date"].dt.date
    q["last_date"] = q["last_date"].dt.date
    q["zero_volume_share"] = (q["zero_volume_days"] / q["rows"]).round(3)
    missing = set(cfg["stocks"]) - set(q.index)
    if missing:
        log.warning("No rows found for: %s - check codes/names in your files", sorted(missing))
    log.info("Cleaning: %d raw rows -> %d rows for %d stocks", n0, len(out), q.shape[0])
    return out, q.reset_index()
