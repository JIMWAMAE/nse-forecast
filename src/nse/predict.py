"""Load saved artifacts and produce next-trading-day forecasts.
Used by the API and by the pipeline's final step."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .clean import prepare_series
from .features import build_stock_features

DISCLAIMER = ("Statistical forecast for research and comparison only. "
              "Not investment advice; prices can move outside the stated range.")


class Forecaster:
    def __init__(self, models_dir: str | Path):
        d = Path(models_dir)
        self.meta = json.loads((d / "metadata.json").read_text())
        self.cfg = self.meta["config"]
        self.egarch = json.loads((d / "egarch_params.json").read_text())
        self.base = xgb.XGBRegressor()
        self.base.load_model(d / "xgb_base.json")
        self.hybrid = xgb.XGBRegressor()
        self.hybrid.load_model(d / "xgb_hybrid.json")
        self.history = pd.read_csv(d / "history.csv", parse_dates=["date"])
        self.stocks = self.cfg["stocks"]

    def _features_last_row(self, code: str, rows: pd.DataFrame) -> pd.Series:
        g = prepare_series(rows.assign(code=code), self.cfg)
        f = build_stock_features(g, self.egarch[code], self.cfg)
        for c in self.stocks:
            f[f"stock_{c}"] = float(c == code)
        return f.iloc[-1]

    def forecast(self, code: str, rows: pd.DataFrame | None = None) -> dict:
        code = code.upper()
        if code not in self.stocks:
            raise KeyError(code)
        if rows is None:
            rows = self.history[self.history["code"] == code]
        last = self._features_last_row(code, rows)
        xb = pd.DataFrame([last[self.meta["features_base"]].astype(float)])
        xh = pd.DataFrame([last[self.meta["features_hybrid"]].astype(float)])
        r_base = float(self.base.predict(xb)[0])
        r_hyb = float(self.hybrid.predict(xh)[0])
        sigma = float(last["egarch_sigma"])
        price = float(last["price"])
        return {
            "code": code,
            "name": self.stocks[code],
            "as_of_date": pd.Timestamp(last["date"]).date().isoformat(),
            "last_price": round(price, 4),
            "forecast_return_pct": round(r_hyb * 100, 4),
            "forecast_price": round(price * np.exp(r_hyb), 4),
            "volatility_1d_pct": round(sigma * 100, 4),
            "range_95_low": round(price * np.exp(r_hyb - 1.96 * sigma), 4),
            "range_95_high": round(price * np.exp(r_hyb + 1.96 * sigma), 4),
            "base_model_price": round(price * np.exp(r_base), 4),
            "naive_price": round(price, 4),
        }

    def forecast_all(self) -> list[dict]:
        out = [self.forecast(c) for c in self.stocks]
        return sorted(out, key=lambda d: d["volatility_1d_pct"], reverse=True)
