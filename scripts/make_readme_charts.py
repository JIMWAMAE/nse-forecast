"""
Builds the stock charts and the stock-summary table used in README.md.
Run from the project root:   .venv/bin/python scripts/make_readme_charts.py
Outputs: docs/images/*.png and an updated stats table inside README.md
"""
import glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("data/raw")
IMG = Path("docs/images"); IMG.mkdir(parents=True, exist_ok=True)
README = Path("README.md")
# Same list as in app.py. None = every stock with at least 1,000 trading days.
TICKERS = None
COLORS = plt.cm.tab10.colors

COLMAP = {"date": "date", "code": "code", "name": "name", "daylow": "day_low",
          "dayhigh": "day_high", "dayprice": "price", "volume": "volume"}

def num(s):
    if s.dtype.kind in "if": return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("%", "").str.strip()
                         .replace({"-": None, "": None}), errors="coerce")

files = [f for f in glob.glob(str(DATA_DIR / "*"))
         if f.lower().endswith((".xlsx", ".xls", ".csv")) and not Path(f).name.startswith("~$")]
assert files, f"No data files in {DATA_DIR.resolve()}"
frames = []
for f in files:
    d = pd.read_csv(f) if f.lower().endswith(".csv") else pd.read_excel(f)
    d.columns = [COLMAP.get(re.sub(r"[^a-z0-9%]", "", str(c).lower()), str(c)) for c in d.columns]
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df["code"] = df["code"].astype(str).str.strip().str.upper()
for c in ["day_low", "day_high", "price", "volume"]:
    if c in df: df[c] = num(df[c])
df = (df.dropna(subset=["date", "price"]).query("price > 0")
        .sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last"))
if TICKERS:
    df = df[df["code"].isin(TICKERS)]
else:
    n = df.groupby("code")["date"].nunique(); df = df[df["code"].isin(n[n >= 1000].index)]
df["ret"] = df.groupby("code")["price"].transform(lambda p: np.log(p).diff())
df.loc[df["ret"].abs() > 0.15, "ret"] = np.nan          # splits / bonus issues
df["year"] = df["date"].dt.year
TICKS = sorted(df["code"].unique())
NAMES = df.groupby("code")["name"].last()
px = df.pivot(index="date", columns="code", values="price").sort_index()
ret = df.pivot(index="date", columns="code", values="ret").sort_index()
cols = 2 if len(TICKS) <= 4 else 5
rows = int(np.ceil(len(TICKS) / cols))

def save(name):
    plt.tight_layout(); plt.savefig(IMG / name, dpi=120, bbox_inches="tight"); plt.close()

# 1. Price history per stock, with 50- and 200-day averages
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), sharex=True, squeeze=False)
for i, (ax, t) in enumerate(zip(axes.flat, TICKS)):
    s = px[t].dropna()
    ax.plot(s, lw=0.9, color=COLORS[i % 10])
    ax.plot(s.rolling(50).mean(), lw=0.8, color="grey", ls="--")
    ax.plot(s.rolling(200).mean(), lw=0.8, color="black")
    ax.set_title(f"{t}", fontsize=10, weight="bold"); ax.tick_params(labelsize=7)
for ax in list(axes.flat)[len(TICKS):]: ax.set_visible(False)
fig.suptitle("Daily closing price (KES), with 50-day (dashed) and 200-day (black) averages", y=1.0)
save("01_price_history.png")

# 2. All stocks rebased to 100
reb = px.apply(lambda s: s / s.dropna().iloc[0] * 100)
fig, ax = plt.subplots(figsize=(13, 5.5))
for i, t in enumerate(TICKS): ax.plot(reb[t], lw=1.1, color=COLORS[i % 10], label=t)
ax.axhline(100, color="k", lw=0.6, ls="--"); ax.set_yscale("log")
ax.set_title("Growth of KES 100 invested at the start (log scale)"); ax.legend(ncol=5, fontsize=8)
save("02_rebased.png")

# 3. Average price per year
avg = df.pivot_table(index="year", columns="code", values="price", aggfunc="mean")
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False)
for i, (ax, t) in enumerate(zip(axes.flat, TICKS)):
    s = avg[t].dropna(); ax.bar(s.index, s.values, color=COLORS[i % 10], alpha=0.8)
    ax.set_title(t, fontsize=10, weight="bold"); ax.tick_params(axis="x", rotation=45, labelsize=7)
for ax in list(axes.flat)[len(TICKS):]: ax.set_visible(False)
fig.suptitle("Average closing price per year (KES)", y=1.0)
save("03_avg_price_by_year.png")

# 4. Oscillation: yearly low-to-high price band
lo = df.pivot_table(index="year", columns="code", values="price", aggfunc="min")
hi = df.pivot_table(index="year", columns="code", values="price", aggfunc="max")
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False)
for i, (ax, t) in enumerate(zip(axes.flat, TICKS)):
    ax.fill_between(lo.index, lo[t], hi[t], color=COLORS[i % 10], alpha=0.25)
    ax.plot(avg.index, avg[t], "o-", color=COLORS[i % 10], ms=3)
    ax.set_title(t, fontsize=10, weight="bold"); ax.tick_params(labelsize=7)
for ax in list(axes.flat)[len(TICKS):]: ax.set_visible(False)
fig.suptitle("How far each stock swung each year: lowest-to-highest price (band) and average (line)", y=1.0)
save("04_yearly_swing.png")

# 5. Rolling volatility (how much prices oscillate day to day)
vol = ret.rolling(63, min_periods=40).std() * np.sqrt(252) * 100
fig, ax = plt.subplots(figsize=(13, 5))
for i, t in enumerate(TICKS): ax.plot(vol[t], lw=1, color=COLORS[i % 10], label=t)
ax.set_ylabel("% per year"); ax.legend(ncol=5, fontsize=8)
ax.set_title("3-month rolling volatility (annualised): higher = bigger price swings")
save("05_rolling_volatility.png")

# 6. Drawdown: how far below its previous peak each stock has been
dd = (px / px.cummax() - 1) * 100
fig, ax = plt.subplots(figsize=(13, 5))
for i, t in enumerate(TICKS): ax.plot(dd[t], lw=0.9, color=COLORS[i % 10], label=t)
ax.set_ylabel("% below previous high"); ax.legend(ncol=5, fontsize=8)
ax.set_title("Drawdown: distance from each stock's previous all-time high in the data")
save("06_drawdown.png")

# 7. Yearly return heatmap
yr = df.groupby(["year", "code"])["price"].agg(["first", "last"])
yret = ((yr["last"] / yr["first"] - 1) * 100).unstack()
fig, ax = plt.subplots(figsize=(1.1 * len(yret) + 3, 0.5 * len(TICKS) + 2))
im = ax.imshow(yret.T.values, cmap="RdYlGn", vmin=-60, vmax=60, aspect="auto")
ax.set_xticks(range(len(yret))); ax.set_xticklabels(yret.index)
ax.set_yticks(range(len(TICKS))); ax.set_yticklabels(yret.columns)
for (j, i), v in np.ndenumerate(yret.T.values):
    if not np.isnan(v): ax.text(i, j, f"{v:.0f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, label="% return in the year"); ax.set_title("Price return per calendar year (%)")
save("07_yearly_returns.png")

# Stats table written into README between the markers
stats = pd.DataFrame({
    "Name": NAMES,
    "First price": px.apply(lambda s: s.dropna().iloc[0]),
    "Last price": px.apply(lambda s: s.dropna().iloc[-1]),
    "Low": px.min(), "High": px.max(),
    "Total change %": px.apply(lambda s: (s.dropna().iloc[-1] / s.dropna().iloc[0] - 1) * 100),
    "Volatility % p.a.": ret.std() * np.sqrt(252) * 100,
    "Max drawdown %": dd.min(),
    "No-change days %": (ret == 0).sum() / ret.notna().sum() * 100,
}).loc[TICKS]
fmt = lambda v: f"{v:,.2f}" if isinstance(v, (float, np.floating)) else str(v)
lines = ["| Code | " + " | ".join(stats.columns) + " |", "|---" * (len(stats.columns) + 1) + "|"]
lines += [f"| {t} | " + " | ".join(fmt(v) for v in r) + " |" for t, r in stats.iterrows()]
period = f"*Period: {df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}. Prices in KES.*"
table = "\n".join([period, ""] + lines)

if README.exists():
    txt = README.read_text()
    txt = re.sub(r"<!-- STATS:START -->.*?<!-- STATS:END -->",
                 f"<!-- STATS:START -->\n{table}\n<!-- STATS:END -->", txt, flags=re.S)
    README.write_text(txt)
print(f"Saved 7 charts to {IMG.resolve()} and updated the stock table in README.md")
