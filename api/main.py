"""FastAPI service for next-trading-day NSE forecasts.

Start locally:  uvicorn api.main:app --reload
Docs:           http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nse.predict import DISCLAIMER, Forecaster

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parents[1] / "models"))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["fc"] = Forecaster(MODELS_DIR)
    yield
    state.clear()


app = FastAPI(
    title="NSE Price & Volatility Forecaster",
    description="Next-trading-day price and volatility forecasts for ten NSE stocks "
                "(EGARCH volatility + XGBoost returns). " + DISCLAIMER,
    version="1.0.0",
    lifespan=lifespan,
)


class Forecast(BaseModel):
    code: str
    name: str
    as_of_date: str
    last_price: float
    forecast_return_pct: float
    forecast_price: float
    volatility_1d_pct: float
    range_95_low: float
    range_95_high: float
    base_model_price: float
    naive_price: float


class ForecastResponse(BaseModel):
    forecasts: list[Forecast]
    disclaimer: str = DISCLAIMER


class DailyRow(BaseModel):
    date: date
    day_price: float = Field(gt=0)
    previous: float | None = Field(default=None, gt=0)
    day_low: float | None = Field(default=None, gt=0)
    day_high: float | None = Field(default=None, gt=0)
    volume: float = Field(default=0, ge=0)
    high_12m: float | None = Field(default=None, gt=0)
    low_12m: float | None = Field(default=None, gt=0)


class HistoryRequest(BaseModel):
    rows: list[DailyRow] = Field(description="Recent daily rows, oldest or newest first (sorted for you).")


def _fc() -> Forecaster:
    return state["fc"]


@app.get("/health")
def health():
    m = _fc().meta
    return {"status": "ok", "model_version": m["version"], "trained_at": m["trained_at"],
            "data_range": m["data_range"]}


@app.get("/stocks")
def stocks():
    return _fc().stocks


@app.get("/model-info")
def model_info():
    m = _fc().meta
    return {k: m[k] for k in ("version", "trained_at", "data_range", "train_rows",
                              "best_params", "test_metrics", "versions")}


@app.get("/forecast", response_model=ForecastResponse)
def forecast_all():
    """All ten stocks, most volatile first, using the latest stored prices."""
    return {"forecasts": _fc().forecast_all()}


@app.get("/forecast/{code}", response_model=ForecastResponse)
def forecast_one(code: str):
    try:
        return {"forecasts": [_fc().forecast(code)]}
    except KeyError:
        raise HTTPException(404, f"Unknown code '{code}'. See /stocks.")


@app.post("/forecast/{code}", response_model=ForecastResponse)
def forecast_from_data(code: str, req: HistoryRequest):
    """Send your own recent price history for one stock and get the next-day forecast."""
    fc = _fc()
    if code.upper() not in fc.stocks:
        raise HTTPException(404, f"Unknown code '{code}'. See /stocks.")
    min_rows = fc.cfg["serving"]["min_rows"]
    if len(req.rows) < min_rows:
        raise HTTPException(422, f"Send at least {min_rows} daily rows (60+ recommended).")
    df = pd.DataFrame([r.model_dump() for r in req.rows]).rename(columns={"day_price": "price"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date", keep="last")
    df["name"] = fc.stocks[code.upper()]
    try:
        return {"forecasts": [fc.forecast(code, df)]}
    except Exception as exc:
        raise HTTPException(422, f"Could not build features from the rows sent: {exc}")
