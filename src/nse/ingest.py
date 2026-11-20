"""Step 1 - read every yearly NSE file in data/raw and stack them into one table.

Handles .csv, .xlsx and .xls files, inconsistent column names, numbers stored as
text (commas, '%', '-') and dates written either day-first or month-first.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COLUMN_MAP = {
    "year": "year",
    "date": "date",
    "code": "code",
    "name": "name",
    "12m_low": "low_12m",
    "12m_high": "high_12m",
    "day_low": "day_low",
    "day_high": "day_high",
    "day_price": "price",
    "previous": "previous",
    "change": "change",
    "change_pct": "change_pct",
    "volume": "volume",
}
NUMERIC = ["low_12m", "high_12m", "day_low", "day_high", "price", "previous",
           "change", "change_pct", "volume"]


def normalise_column(col: str) -> str:
    c = str(col).strip().lower().replace("%", "_pct")
    return re.sub(r"[^a-z0-9]+", "_", c).strip("_")


def to_number(s: pd.Series) -> pd.Series:
    """'1,234.5' -> 1234.5 ; '-' / '' -> NaN ; '2.5%' -> 2.5"""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = (s.astype(str).str.strip()
               .str.replace(",", "", regex=False)
               .str.replace("%", "", regex=False)
               .replace({"-": np.nan, "": np.nan, "nan": np.nan, "None": np.nan}))
    return pd.to_numeric(cleaned, errors="coerce")


def _order_score(dates: pd.Series) -> float:
    """Share of consecutive rows (file order) that are within 7 days of each other.
    The correct day/month reading gives a smooth sequence; the wrong one jumps around."""
    d = dates.dropna()
    if len(d) < 3:
        return 0.0
    gaps = d.diff().abs().dt.days.dropna()
    return float((gaps <= 7).mean())


def parse_dates(raw: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(raw):
        return raw
    s = raw.astype(str).str.strip()
    candidates = []
    for dayfirst in (True, False):
        parsed = pd.to_datetime(s, dayfirst=dayfirst, format="mixed", errors="coerce")
        candidates.append((parsed.isna().sum(), -_order_score(parsed), dayfirst, parsed))
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[0][3]


def read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=str, encoding="latin-1")
    else:
        df = pd.read_excel(path)
    df.columns = [normalise_column(c) for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)
    missing = {"date", "code", "price"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}. Found {list(df.columns)}")
    keep = [c for c in COLUMN_MAP.values() if c in df.columns]
    df = df[keep].copy()
    df["date"] = parse_dates(df["date"])
    for c in NUMERIC:
        if c in df.columns:
            df[c] = to_number(df[c])
    df["code"] = df["code"].astype(str).str.strip().str.upper()
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
    df["source_file"] = path.name
    return df


def ingest(raw_dir: Path) -> pd.DataFrame:
    files = sorted(p for p in Path(raw_dir).iterdir()
                   if p.suffix.lower() in {".csv", ".xlsx", ".xls"} and not p.name.startswith("~$"))
    if not files:
        raise FileNotFoundError(f"No .csv/.xlsx files found in {raw_dir}. Put the yearly NSE files there.")
    frames = []
    for f in files:
        df = read_file(f)
        log.info("Read %-40s %7d rows  %s -> %s", f.name, len(df),
                 df["date"].min().date() if df["date"].notna().any() else "?",
                 df["date"].max().date() if df["date"].notna().any() else "?")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for c in NUMERIC:
        if c not in out.columns:
            out[c] = np.nan
    return out
