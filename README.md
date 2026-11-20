# Forecasting NSE Stock Prices with a Hybrid EGARCH–XGBoost Model

A capstone project that forecasts daily prices of selected **Nairobi Securities Exchange (NSE)** stocks. It combines an econometric volatility model (**EGARCH**) with a machine-learning model (**XGBoost**), then deploys the result as a simple web app in **Docker**.

> ⚠️ **Disclaimer:** This project is for academic research only. It is **not financial advice**. Stock prices are uncertain and can move outside any forecast range shown here.

---

## Contents
1. [Project overview](#1-project-overview)
2. [The stocks](#2-the-stocks)
3. [How prices have moved](#3-how-prices-have-moved)
4. [Data](#4-data)
5. [Method](#5-method)
6. [Results](#6-results)
7. [The forecast app](#7-the-forecast-app)
8. [How to run](#8-how-to-run)
9. [Project structure](#9-project-structure)
10. [Limitations and next steps](#10-limitations-and-next-steps)

---

## 1. Project overview

**Problem.** NSE stocks are hard to forecast. Many trade thinly, prices jump suddenly, and calm periods alternate with stormy ones. A single model rarely handles all of this well.

**Approach.**
- **EGARCH(1,1,1)** with Student-t errors models **volatility**, meaning how much a price is likely to move. It also captures the fact that bad news tends to raise volatility more than good news.
- **XGBoost** learns patterns from **technical features** such as recent returns, momentum, volume and trading range.
- The **hybrid** model feeds EGARCH's volatility estimates into XGBoost as extra inputs.

**Objectives.**
1. Prepare and explore daily NSE data for the selected stocks.
2. Build EGARCH, XGBoost and hybrid models.
3. Compare them against simple baselines on unseen data.
4. Deploy the hybrid model as an easy-to-use forecast app.

---

## 2. The stocks

| Code | Company | Sector |
|---|---|---|
| ABSA | ABSA Bank Kenya Plc | Banking |
| BRIT | Britam Holdings Plc | Insurance |
| COOP | Co-operative Bank of Kenya Ltd | Banking |
| EQTY | Equity Group Holdings Plc | Banking |
| KCB | KCB Group Plc | Banking |
| KQ | Kenya Airways Plc | Commercial and services (aviation) |
| KEGN | Kenya Electricity Generating Company Plc (KenGen) | Energy |
| KPLC | Kenya Power and Lighting Company Plc | Energy |
| _add_ | _stock 9_ | _sector_ |
| _add_ | _stock 10_ | _sector_ |

**Why these stocks:** they cover the NSE's main sectors (banking, insurance, energy and aviation) and mix heavily traded, steady stocks with thinly traded, volatile ones. That mix is a good test for the models.

### Key statistics
The table below is filled in automatically from the data by `make_readme_charts.py` (see section 8).

<!-- STATS:START -->
*Run `make_readme_charts.py` to fill in this table.*
<!-- STATS:END -->

**How to read it:**
- **Volatility % p.a.** shows how much the price typically swings in a year. Higher means riskier.
- **Max drawdown %** is the biggest fall from a previous high.
- **No-change days %** is the share of days the price didn't move at all, a sign of thin trading.

---

## 3. How prices have moved

### 3.1 Price history
Each stock's daily closing price, with its 50-day and 200-day moving averages. When the price crosses its averages, that marks a change in trend.

![Price history](docs/images/01_price_history.png)

### 3.2 Relative performance
All stocks start at 100, so you can compare them on the same scale regardless of their share price.

![Rebased performance](docs/images/02_rebased.png)

### 3.3 Average price per year
The yearly average smooths out daily noise and shows the longer-term direction of each stock.

![Average price by year](docs/images/03_avg_price_by_year.png)

### 3.4 How far prices swing each year
The shaded band runs from each year's lowest to highest price, and the line is the yearly average. A wide band means the stock oscillated a lot that year.

![Yearly swing](docs/images/04_yearly_swing.png)

### 3.5 Volatility over time
The 3-month rolling volatility shows that calm and stormy periods **cluster** together. This is the main reason for using EGARCH.

![Rolling volatility](docs/images/05_rolling_volatility.png)

### 3.6 Drawdowns
This chart shows how far each stock has been below its previous high. It shows how deep and how long each stock's losing periods were.

![Drawdown](docs/images/06_drawdown.png)

### 3.7 Return per calendar year
Green years are gains and red years are losses. The heatmap makes good and bad years easy to compare across stocks.

![Yearly returns](docs/images/07_yearly_returns.png)

### 3.8 What this tells us
- **Price levels and trends differ a lot** between stocks, so the models use **returns**, not raw prices.
- **Volatility clusters:** big moves follow big moves. EGARCH is built for this.
- **Returns have fat tails:** extreme days happen more often than a normal curve predicts, so EGARCH uses Student-t errors.
- **Thin trading matters:** many no-change days create short-term patterns that XGBoost can pick up.

---

## 4. Data

**Source:** daily equity trading data from the Nairobi Securities Exchange. *(Add the exact provider.)*
**Period:** January 2017 to June 2026, trading days only.
**Format:** one Excel file per year, stored in `data/raw/`. Raw data is not included in this repository. See section 8.

| Variable | Description |
|---|---|
| `DATE` | Trading date |
| `CODE` / `NAME` | Stock ticker and company name |
| `Day Price` | Closing price (KES), the **main variable** |
| `Day Low` / `Day High` | Lowest and highest traded price that day |
| `Previous` | Previous day's closing price |
| `Change` / `Change%` | Change from the previous close |
| `12m Low` / `12m High` | Lowest and highest price over the past 12 months |
| `Volume` | Shares traded that day |

**Cleaning:** unreadable dates, duplicates and zero prices are removed. Daily moves above 15% (usually stock splits, bonus issues or data errors) are excluded from modelling.

---

## 5. Method

### 5.1 Time split
| Set | Period | Use |
|---|---|---|
| Training | 2017 – 2023 | Fit the models |
| Validation | 2024 – 2025 | Tune the settings |
| Test | Jan – Jun 2026 | Final evaluation on unseen data |

The data is split by date and never shuffled, so no model sees the future.

### 5.2 Models
| Model | What it does |
|---|---|
| Random walk | Baseline: predicts no change tomorrow |
| Historical mean | Baseline: predicts the stock's average daily return |
| EGARCH(1,1,1) | One model per stock: volatility with a leverage effect, Student-t errors |
| XGBoost | One model across all stocks, using about 20 technical features |
| **Hybrid** | XGBoost + EGARCH volatility, mean and shock as extra features |

### 5.3 Evaluation
- **RMSE and MAE** of the next-day return
- **Directional accuracy:** how often the up/down call is right
- **Price MAPE:** the error in the forecast price
- **Diebold-Mariano test:** whether differences between models are statistically significant

---

## 6. Results

*Fill in from the notebook's model comparison table (section 5.7) after running on your data.*

| Model | Test RMSE (%) | Test MAE (%) | Directional accuracy | Price MAPE (%) |
|---|---|---|---|---|
| Random walk | | | – | |
| Historical mean | | | | |
| EGARCH | | | | |
| XGBoost | | | | |
| **Hybrid** | | | | |

**Key findings:** *(summarise 3 to 4 points, for example which model did best, which stocks were easiest or hardest to forecast, and what the EGARCH parameters showed.)*

---

## 7. The forecast app

A one-page **Streamlit** app, packaged in **Docker**.

**How to use it:**
1. Pick a **stock** from the drop-down.
2. Pick a **forecast period:** 1 day, 1 week, 2 weeks, 1 month or 3 months.
3. Click **Forecast**.

**What you get:**
- The last price, the forecast price with its % change, and an 80% likely range
- A chart of the last 90 days, with the forecast as a **dotted line** and a shaded range
- A daily forecast table
- A disclaimer

**How the forecast is built:**
- **Day 1** comes from the hybrid model.
- **After that,** the signal fades toward the stock's long-run trend.
- **The range** comes from EGARCH volatility, capped relative to each stock's normal level. This keeps ranges realistic for volatile, thinly traded stocks.

```
data/raw (Excel)  →  app.py  →  EGARCH + Hybrid XGBoost  →  web page (port 8501)
                        └──────── Docker container ────────┘
```

---

## 8. How to run

### 8.1 Requirements
- macOS, Linux or Windows
- Python 3.12
- Docker Desktop (optional, for the container)

### 8.2 Set up
```bash
git clone https://github.com/<your-username>/nse-forecast.git
cd nse-forecast
make setup PY=python3.12
```
Then copy your NSE Excel files into `data/raw/`.

### 8.3 Common commands
| Task | Command |
|---|---|
| Run the full model pipeline | `make pipeline` |
| Run the tests | `make test` |
| Make the README charts and stats table | `.venv/bin/python make_readme_charts.py` |
| Open the notebook | open `nse_capstone_notebook.ipynb` in VS Code or Jupyter |
| Run the app locally | `.venv/bin/streamlit run app.py` |
| Run the app in Docker | `docker build -f Dockerfile.app -t nse-app .` then `docker run --rm -p 8501:8501 nse-app` |

Once the app is running, open **http://localhost:8501** in your browser.

---

## 9. Project structure
```
nse-forecast/
├── app.py                      # Streamlit forecast app
├── Dockerfile.app              # Container for the app
├── requirements-app.txt        # App dependencies
├── Dockerfile / Makefile       # Pipeline, API and tooling
├── make_readme_charts.py       # Builds the charts and table in this README
├── nse_capstone_notebook.ipynb # EDA, modelling and deployment notebook
├── data/raw/                   # NSE Excel files (not in the repository)
├── docs/images/                # README charts
└── reports/figures/            # Notebook charts
```

---

## 10. Limitations and next steps
- EGARCH parameters are fixed after training. Re-estimating them monthly may improve the forecasts.
- Daily returns are close to unpredictable, so the hybrid's advantage is mainly **one day ahead**. Longer forecasts rely mostly on volatility and the long-run trend.
- Thinly traded stocks have many no-change days, so some apparent patterns may not be tradable after costs.
- **Next steps:** an automatic daily data refresh, quantile models for forecast ranges, SHAP explanations, and cloud hosting.

---

**Author:** Jimcollins Wamae Wanguku · Capstone project, 2026
