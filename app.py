"""
NSE Stock Forecaster: a simple one-page app.
Run:  streamlit run app.py
"""
import glob, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from arch import arch_model

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- settings
DATA_DIR = next((p for p in [Path("data/raw"), Path("../data/raw")] if p.exists()), Path("data/raw"))
HISTORY_DAYS = 90          # days of actual prices shown on the chart
# Put your 10 stock codes here, e.g. ["ABSA", "BRIT", "COOP", "EQTY", "KCB", "KQ", "KEGN", "KPLC", ...]
# Leave as None to show every stock with enough history.
TICKERS = None
HORIZONS = {"1 day": 1, "1 week (5 days)": 5, "2 weeks (10 days)": 10,
            "1 month (21 days)": 21, "3 months (63 days)": 63}
BAND_Z = 1.28              # 80% likely range

# ---------------------------------------------------------------- data
COLMAP = {"year": "year", "date": "date", "code": "code", "name": "name",
          "12mlow": "low_12m", "12mhigh": "high_12m", "daylow": "day_low", "dayhigh": "day_high",
          "dayprice": "price", "previous": "previous", "change": "change",
          "change%": "change_pct", "volume": "volume"}


def _num(s):
    if s.dtype.kind in "if":
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("%", "").str.strip()
                         .replace({"-": None, "": None}), errors="coerce")


@st.cache_data(show_spinner="Loading NSE data...")
def load_data():
    files = [f for f in glob.glob(str(DATA_DIR / "*"))
             if f.lower().endswith((".xlsx", ".xls", ".csv")) and not Path(f).name.startswith("~$")]
    if not files:
        return None
    frames = []
    for f in files:
        d = pd.read_csv(f) if f.lower().endswith(".csv") else pd.read_excel(f)
        d.columns = [COLMAP.get(re.sub(r"[^a-z0-9%]", "", str(c).lower()), str(c)) for c in d.columns]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["code"] = df["code"].astype(str).str.strip().str.upper()
    df["name"] = df["name"].astype(str).str.strip()
    for c in ["day_low", "day_high", "price", "volume"]:
        if c in df:
            df[c] = _num(df[c])
    df = (df.dropna(subset=["date", "price"]).query("price > 0")
            .sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last"))
    if TICKERS:
        return df[df["code"].isin(TICKERS)]
    counts = df.groupby("code")["date"].nunique()
    return df[df["code"].isin(counts[counts >= 500].index)]


# ---------------------------------------------------------------- model
def _features(s):
    """One stock's rows indexed by date. Each feature uses data up to that day only."""
    f = pd.DataFrame(index=s.index)
    r = np.log(s["price"]).diff()
    r[r.abs() > 0.15] = np.nan                       # ignore splits / bonus issues / data errors
    for k in range(1, 6):
        f[f"lag{k}"] = r.shift(k - 1)
    f["ma5"], f["ma21"] = r.rolling(5, min_periods=3).mean(), r.rolling(21, min_periods=10).mean()
    f["vol5"], f["vol21"] = r.rolling(5, min_periods=3).std(), r.rolling(21, min_periods=10).std()
    f["range"] = (s["day_high"] - s["day_low"]) / s["price"]
    f["gap20"] = s["price"] / s["price"].rolling(20).mean() - 1
    lv = np.log1p(s["volume"].fillna(0))
    f["vol_z"] = (lv - lv.rolling(63, min_periods=20).mean()) / lv.rolling(63, min_periods=20).std()
    return f.replace([np.inf, -np.inf], np.nan), r


@st.cache_data(show_spinner="Running the forecast...")
def forecast(code, horizon):
    df = load_data()
    s = df[df["code"] == code].set_index("date")
    feats, r = _features(s)
    rc = r.dropna()

    # 1) EGARCH(1,1,1) with Student-t errors, on clean returns in %
    res = arch_model((rc * 100).clip(-10, 10), mean="Constant", vol="EGARCH", p=1, o=1, q=1,
                     dist="t", rescale=False).fit(disp="off", options={"maxiter": 500})
    feats["eg_vol"] = (res.conditional_volatility / 100).reindex(feats.index)

    # Volatility path: start at tomorrow's EGARCH volatility, then fade back to the stock's normal
    # (last 12 months) level. This avoids unstable multi-step simulations on thinly traded stocks.
    vol_now = float(res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]) ** 0.5 / 100
    vol_long = float(rc.tail(252).std())
    vol_now = float(np.clip(vol_now, 0.25 * vol_long, 3 * vol_long))
    beta = float(np.clip(res.params.filter(like="beta").iloc[0], 0.5, 0.98))
    k = np.arange(horizon)
    daily_vol = np.sqrt(vol_long ** 2 + (vol_now ** 2 - vol_long ** 2) * beta ** k)

    # 2) Hybrid XGBoost (technical features + EGARCH volatility) for the next-day return
    data = feats.assign(target=r.shift(-1))
    train = data.dropna(subset=["target"]).dropna(thresh=len(feats.columns) - 2)
    model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, objective="reg:pseudohubererror", huber_slope=0.02,
                             random_state=42)
    model.fit(train[feats.columns], train["target"])
    next_ret = float(model.predict(feats[feats.columns].iloc[[-1]])[0])
    next_ret = float(np.clip(next_ret, -2 * vol_now, 2 * vol_now))

    # 3) Price path: day 1 from the hybrid model; the signal then fades toward a small long-run drift
    drift = float(np.clip(rc.tail(252).mean(), -0.002, 0.002))
    rets = drift + (next_ret - drift) * 0.5 ** k
    cum_ret = np.cumsum(rets)
    cum_sd = np.sqrt(np.cumsum(daily_vol ** 2))

    last_px, last_date = float(s["price"].iloc[-1]), s.index[-1]
    dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
    path = pd.DataFrame({"date": dates,
                         "price": last_px * np.exp(cum_ret),
                         "low": last_px * np.exp(cum_ret - BAND_Z * cum_sd),
                         "high": last_px * np.exp(cum_ret + BAND_Z * cum_sd)})
    hist = s["price"].tail(HISTORY_DAYS)
    return hist, path, last_px, last_date, s["name"].iloc[-1]


# ---------------------------------------------------------------- page
st.set_page_config(page_title="NSE Stock Forecaster", page_icon="📈", layout="centered")
st.markdown("<style>#MainMenu, footer, header {visibility:hidden;} .block-container{padding-top:2rem;}</style>",
            unsafe_allow_html=True)

st.title("📈 NSE Stock Forecaster")
st.caption("Pick a stock and see where the model expects its price to go.")

df = load_data()
if df is None or df.empty:
    st.error(f"No data found. Put your NSE .xlsx or .csv files in **{DATA_DIR.resolve()}** and refresh.")
    st.stop()

names = df.groupby("code")["name"].last()
c1, c2 = st.columns([2, 1])
code = c1.selectbox("Stock", names.index, format_func=lambda c: f"{names[c]} ({c})")
h_label = c2.selectbox("Forecast for", list(HORIZONS), index=1)
horizon = HORIZONS[h_label]

if st.button("Forecast", type="primary", width="stretch"):
    try:
        hist, path, last_px, last_date, name = forecast(code, horizon)
    except Exception as e:
        st.error(f"Could not run the forecast for {code}: {e}")
        st.stop()

    end_px = path["price"].iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Last price ({last_date:%d %b %Y})", f"KES {last_px:,.2f}")
    m2.metric(f"Forecast ({h_label})", f"KES {end_px:,.2f}", f"{end_px / last_px - 1:+.2%}")
    m3.metric("Likely range (80%)", f"{path['low'].iloc[-1]:,.2f} – {path['high'].iloc[-1]:,.2f}")

    fx = [last_date] + list(path["date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fx + fx[::-1],
                             y=[last_px] + list(path["high"]) + (list(path["low"]) + [last_px])[::-1],
                             fill="toself", fillcolor="rgba(231,111,81,0.15)", line=dict(width=0),
                             hoverinfo="skip", name="Likely range (80%)"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Actual price",
                             line=dict(color="#264653", width=2)))
    fig.add_trace(go.Scatter(x=fx, y=[last_px] + list(path["price"]), mode="lines+markers",
                             name="Forecast", line=dict(color="#e76f51", width=2.5, dash="dot"),
                             marker=dict(size=4)))
    fig.update_layout(title=f"{name}: last {HISTORY_DAYS} trading days and forecast",
                      yaxis_title="Price (KES)", hovermode="x unified", height=440,
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Daily forecast"):
        st.table(path.assign(date=path["date"].dt.strftime("%a %d %b %Y"))
                     .rename(columns={"date": "Date", "price": "Forecast (KES)",
                                      "low": "Low (KES)", "high": "High (KES)"})
                     .round(2).set_index("Date"))

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This forecast comes from a statistical model (EGARCH + XGBoost) built for academic "
    "research. It is not financial advice or a recommendation to buy or sell any security. The likely range "
    "covers about 8 in 10 outcomes, so the real price can move outside it. Longer forecasts are less certain. "
    "Past performance does not guarantee future results. Talk to a licensed investment adviser before investing."
)
