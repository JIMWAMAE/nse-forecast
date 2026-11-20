"""Create SYNTHETIC NSE-style yearly files for smoke tests and CI.

NOT real market data. It copies the real file layout (same columns, text numbers,
mixed date formats, old BBK code, duplicates, '-' prices, a bonus issue) so the
pipeline can be tested without the Mendeley files.

    python scripts/make_sample_data.py --out data/sample_raw --start 2017 --end 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STOCKS = {
    "ABSA": ("ABSA Bank Kenya Plc", 12.0), "BRIT": ("Britam Holdings Plc", 9.0),
    "COOP": ("Co-operative Bank of Kenya Ltd", 14.0), "EQTY": ("Equity Group Holdings Plc", 40.0),
    "KCB": ("KCB Group Plc", 45.0), "KQ": ("Kenya Airways Plc", 6.0),
    "KEGN": ("Kenya Electricity Generating Company Plc", 7.0),
    "KPLC": ("Kenya Power and Lighting Company Plc", 3.0),
    "KNRE": ("Kenya Re-Insurance Corporation Ltd", 2.0), "SCOM": ("Safaricom Plc", 25.0),
    "XYZ": ("Some Other Company Plc", 10.0),
}


def simulate(start: int, end: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(f"{start}-01-01", f"{end}-06-30")
    rows = []
    for code, (name, p0) in STOCKS.items():
        price, sig = p0, 0.015
        hist: list[float] = []
        for i, d in enumerate(days):
            sig = np.sqrt(0.00002 + 0.08 * (hist[-1] if hist else 0) ** 2 + 0.88 * sig ** 2)
            r = rng.standard_t(5) * sig * 0.8
            r = float(np.clip(r, -0.09, 0.09))
            prev = price
            if code == "KNRE" and d == pd.Timestamp("2019-06-03"):
                prev = price / 1.5   # 1-for-2 bonus: exchange adjusts 'Previous'
            price = round(max(prev * np.exp(r), 0.2), 2)
            hist.append(np.log(price / prev))
            vol = 0 if rng.random() < (0.15 if code in {"KNRE", "KPLC"} else 0.02) else int(rng.lognormal(11, 1.2))
            if vol == 0:
                price = round(prev, 2)
            lo = round(min(price, prev) * (1 - abs(rng.normal(0, sig / 2))), 2)
            hi = round(max(price, prev) * (1 + abs(rng.normal(0, sig / 2))), 2)
            rows.append(dict(DATE=d, CODE=("BBK" if code == "ABSA" and d.year < 2020 else code),
                             NAME=("Barclays Bank Ltd" if code == "ABSA" and d.year < 2020 else name),
                             day_low=lo, day_high=hi, price=price, prev=round(prev, 2), vol=vol))
    df = pd.DataFrame(rows)
    df = df.sort_values(["DATE", "CODE"])
    g = df.groupby("CODE")
    df["12m Low"] = g["day_low"].transform(lambda s: s.rolling(250, min_periods=1).min())
    df["12m High"] = g["day_high"].transform(lambda s: s.rolling(250, min_periods=1).max())
    return df


def write(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for year, y in df.groupby(df["DATE"].dt.year):
        fmt = "%d/%m/%Y" if year % 2 else "%m/%d/%Y"   # mixed date conventions on purpose
        o = pd.DataFrame({
            "YEAR": year, "DATE": y["DATE"].dt.strftime(fmt), "CODE": y["CODE"], "NAME": y["NAME"],
            "12m Low": y["12m Low"].round(2), "12m High": y["12m High"].round(2),
            "Day Low": y["day_low"], "Day High": y["day_high"], "Day Price": y["price"].astype(object),
            "Previous": y["prev"], "Change": (y["price"] - y["prev"]).round(2),
            "Change%": ((y["price"] / y["prev"] - 1) * 100).round(2).astype(str) + "%",
            "Volume": y["vol"].map(lambda v: f"{v:,}"),
        })
        o.loc[o.sample(frac=0.002, random_state=year).index, "Day Price"] = "-"
        o = pd.concat([o, o.sample(5, random_state=year)])  # duplicates
        o.to_csv(out / f"NSE_data_all_stocks_{year}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample_raw")
    ap.add_argument("--start", type=int, default=2017)
    ap.add_argument("--end", type=int, default=2026)
    a = ap.parse_args()
    write(simulate(a.start, a.end), Path(a.out))
    print(f"Synthetic files written to {a.out}")
