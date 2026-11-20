# Report / Slide Guide

Fill the numbers from `reports/RESULTS.md` after running `make pipeline` on the real data.
The rubric weights are in brackets.

## 1. Problem framing  [Report 10%]
- Mobile platforms such as Ziidi Trader make NSE investing accessible, but new investors lack a
  view of likely next-day moves and how uncertain they are.
- Task: forecast next-trading-day price and volatility for ten high-volume stocks; test whether
  a volatility forecast improves price prediction.

## 2. Data pipeline  [10%]
- Source, period, rows per stock (`data_quality.csv`).
- Issues found and how each was handled: mixed date formats, text numbers, BBK→ABSA,
  duplicates, missing prices, zero-volume days, price adjustments via `Previous`, extreme moves.
- Leakage controls: features use data up to day t only; EGARCH parameters fit on the training
  window only; splits by the target date; automated leakage test (`tests/test_leakage.py`).

## 3. Technique justification  [10%]

**Chosen**
| Technique | Why it fits |
|---|---|
| Log returns, not prices | Prices trend and are non-stationary; returns are comparable across stocks with very different price levels (KNRE ~KES 2 vs EQTY ~KES 60). |
| EGARCH(1,1), Student-t | Volatility clusters and reacts more to bad news than good (leverage effect); EGARCH models this without positivity constraints; t errors handle fat tails. |
| XGBoost | Captures non-linear links between returns, volume, ranges and volatility; robust to missing values; works on ~23k pooled rows; fast to tune. |
| One pooled model with stock indicators | ~2,400 rows per stock is small; pooling shares patterns across stocks while stock dummies keep differences. |
| Base vs hybrid vs naive | Isolates the value of the volatility input; the naive (last price) baseline is the standard bar in finance. |
| Chronological validation | Random K-fold would leak future information in time series. |
| Diebold–Mariano test | Checks whether error differences are statistically meaningful, not chance. |
| QLIKE for volatility | Fair loss when true volatility is only seen through noisy squared returns. |

**Considered and ruled out**
| Alternative | Why not |
|---|---|
| LSTM / Transformers | Too little data per stock for deep sequence models; harder to explain; studies on frontier markets show no consistent gain over boosted trees. |
| ARIMA for prices | Linear and assumes constant variance; used implicitly by the naive random-walk benchmark. |
| Plain GARCH | Symmetric response to good and bad news. EGARCH nests this idea and adds asymmetry. |
| Prophet | Built for business seasonality, not daily return noise. |
| Random K-fold CV | Leaks the future. |
| Price-level target | Models end up copying yesterday's price; looks accurate but adds nothing. |
| MLflow / drift monitoring | Single-user project with a fixed hold-out; config + metadata.json give traceability. Drift monitoring noted as future work. |

## 4. Results  [Modelling & evaluation 20%]
- Table: overall test metrics (naive vs base vs hybrid).
- Table: by stock MAPE and direction accuracy.
- Diebold–Mariano results: say clearly where the hybrid is or is not better.
- Volatility: EGARCH vs 20-day rolling std (QLIKE).
- Figures: `test_prices.png`, `test_volatility.png`, `importance_hybrid.png`.
- **Interpret, don't just report.** Expect close to random-walk accuracy for prices; the volatility
  forecast and 95% range are often the more useful output. A small or no gain is still a valid
  result if explained.

## 5. Deployment  [10%]
- Architecture diagram (README), API endpoints, Docker commands, sample request/response.

## 6. Limitations and real-world relevance
- Thin trading and stale prices; corporate actions; 6-month test window; no news or macro data.
- Use case: a comparison view of expected move and uncertainty across stocks for new investors.
- Not investment advice.
