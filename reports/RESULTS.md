# Results summary (auto-generated)

Tuning: train to 2023-12-31, validate to 2025-12-31. Final test: 2026-01-01 onwards (used once).

## Data quality
| code   | first_date   | last_date   |   rows |   zero_volume_days |   price_adjustments |   outlier_returns |   dropped_missing_price |   dropped_duplicates |   zero_volume_share |
|:-------|:-------------|:------------|-------:|-------------------:|--------------------:|------------------:|------------------------:|---------------------:|--------------------:|
| ABSA   | 2015-05-09   | 2026-06-30  |   2353 |                  2 |                   4 |                 0 |                       0 |                    1 |               0.001 |
| BRIT   | 2015-05-09   | 2026-06-30  |   2353 |                  1 |                  18 |                 0 |                       0 |                    1 |               0     |
| COOP   | 2015-05-09   | 2026-06-30  |   2353 |                  0 |                   8 |                 0 |                       0 |                    1 |               0     |
| EQTY   | 2015-05-09   | 2026-06-30  |   2353 |                  0 |                   9 |                 0 |                       0 |                    1 |               0     |
| KCB    | 2015-05-09   | 2026-06-30  |   2353 |                  0 |                   8 |                 0 |                       0 |                    1 |               0     |
| KEGN   | 2015-05-09   | 2026-06-30  |   2353 |                  1 |                   2 |                 1 |                       0 |                    1 |               0     |
| KNRE   | 2015-05-09   | 2026-06-30  |   2353 |                  0 |                  11 |                 3 |                       0 |                    1 |               0     |
| KPLC   | 2015-05-09   | 2026-06-30  |   2353 |                  0 |                  14 |                 1 |                       0 |                    1 |               0     |
| KQ     | 2015-05-09   | 2026-06-30  |   2353 |               1126 |                   2 |                 1 |                       0 |                    1 |               0.479 |
| SCOM   | 2015-05-09   | 2026-06-30  |   2353 |                  0 |                  12 |                 0 |                       0 |                    1 |               0     |

## Held-out test - all stocks
| model                     |    n |   ret_rmse |   ret_mae |   price_rmse |   price_mae |   price_mape_pct |   dir_acc |   days_moved |
|:--------------------------|-----:|-----------:|----------:|-------------:|------------:|-----------------:|----------:|-------------:|
| Last price (naive)        | 1220 |    0.02133 |   0.01285 |      0.57385 |     0.29859 |          1.28016 | nan       |         1052 |
| XGBoost (no volatility)   | 1220 |    0.01936 |   0.01224 |      0.52626 |     0.28846 |          1.21907 |   0.60741 |         1052 |
| EGARCH + XGBoost (hybrid) | 1220 |    0.01947 |   0.01227 |      0.52655 |     0.28812 |          1.22317 |   0.61312 |         1052 |

## Held-out test - by stock
| code   |   price_mape_hybrid |   price_mape_base |   price_mape_naive |   dir_acc_hybrid |   dir_acc_base |
|:-------|--------------------:|------------------:|-------------------:|-----------------:|---------------:|
| ABSA   |              1.1921 |            1.1998 |             1.2595 |           0.6    |         0.5909 |
| BRIT   |              1.5933 |            1.5946 |             1.5923 |           0.5429 |         0.5619 |
| COOP   |              1.2245 |            1.2277 |             1.227  |           0.5565 |         0.5565 |
| EQTY   |              0.9533 |            0.9524 |             0.9732 |           0.6304 |         0.6087 |
| KCB    |              0.8221 |            0.8279 |             0.8651 |           0.6966 |         0.6966 |
| KEGN   |              0.6375 |            0.648  |             0.577  |           0.5326 |         0.5    |
| KNRE   |              1.2786 |            1.2934 |             1.3847 |           0.6429 |         0.6429 |
| KPLC   |              1.0164 |            1.0299 |             1.1091 |           0.6239 |         0.6147 |
| KQ     |              2.6327 |            2.5417 |             2.9185 |           0.6695 |         0.661  |
| SCOM   |              0.8813 |            0.8752 |             0.8951 |           0.6364 |         0.6364 |

## Diebold-Mariano (squared return errors; negative = hybrid better)
| code   | hybrid_vs               |   dm_stat |   p_value |
|:-------|:------------------------|----------:|----------:|
| ABSA   | XGBoost (no volatility) |   -0.509  |    0.6117 |
| ABSA   | Last price (naive)      |   -3.234  |    0.0016 |
| BRIT   | XGBoost (no volatility) |   -0.199  |    0.8426 |
| BRIT   | Last price (naive)      |   -1.3033 |    0.195  |
| COOP   | XGBoost (no volatility) |   -0.559  |    0.5772 |
| COOP   | Last price (naive)      |   -1.1108 |    0.2688 |
| EQTY   | XGBoost (no volatility) |   -0.4316 |    0.6668 |
| EQTY   | Last price (naive)      |   -2.5886 |    0.0108 |
| KCB    | XGBoost (no volatility) |    0.4147 |    0.6791 |
| KCB    | Last price (naive)      |   -2.0095 |    0.0467 |
| KEGN   | XGBoost (no volatility) |   -2.2984 |    0.0233 |
| KEGN   | Last price (naive)      |   -0.2403 |    0.8105 |
| KNRE   | XGBoost (no volatility) |   -0.8858 |    0.3775 |
| KNRE   | Last price (naive)      |   -2.0159 |    0.046  |
| KPLC   | XGBoost (no volatility) |   -1.5984 |    0.1126 |
| KPLC   | Last price (naive)      |   -2.671  |    0.0086 |
| KQ     | XGBoost (no volatility) |    1.7403 |    0.0844 |
| KQ     | Last price (naive)      |   -3.0281 |    0.003  |
| SCOM   | XGBoost (no volatility) |    0.1611 |    0.8723 |
| SCOM   | Last price (naive)      |   -2.0508 |    0.0424 |
| ALL    | XGBoost (no volatility) |    0.7975 |    0.4253 |
| ALL    | Last price (naive)      |   -5.2842 |    0      |

## Volatility forecasts (lower is better)
| code   | model              |   var_rmse |   abs_ret_mae |    qlike |
|:-------|:-------------------|-----------:|--------------:|---------:|
| ABSA   | EGARCH(1,1)        |   0.000878 |      0.01406  | -6.92114 |
| ABSA   | 20-day rolling std |   0.000838 |      0.012891 | -6.6734  |
| BRIT   | EGARCH(1,1)        |   0.001296 |      0.016019 | -6.68596 |
| BRIT   | 20-day rolling std |   0.001312 |      0.015026 | -6.53283 |
| COOP   | EGARCH(1,1)        |   0.00076  |      0.014806 | -7.01255 |
| COOP   | 20-day rolling std |   0.000633 |      0.012076 | -6.83216 |
| EQTY   | EGARCH(1,1)        |   0.000986 |      0.015197 | -7.22179 |
| EQTY   | 20-day rolling std |   0.000628 |      0.010393 | -6.27352 |
| KCB    | EGARCH(1,1)        |   0.000938 |      0.018327 | -7.2311  |
| KCB    | 20-day rolling std |   0.000465 |      0.00942  | -7.30854 |
| KEGN   | EGARCH(1,1)        |   0.000485 |      0.014265 | -7.80901 |
| KEGN   | 20-day rolling std |   0.000308 |      0.007218 | -7.30745 |
| KNRE   | EGARCH(1,1)        |   0.001518 |      0.017391 | -6.80632 |
| KNRE   | 20-day rolling std |   0.001477 |      0.015357 | -4.74246 |
| KPLC   | EGARCH(1,1)        |   0.000825 |      0.015978 | -7.09    |
| KPLC   | 20-day rolling std |   0.000601 |      0.010518 | -7.09713 |
| KQ     | EGARCH(1,1)        |   0.002646 |      0.024608 | -5.59038 |
| KQ     | 20-day rolling std |   0.00301  |      0.027776 | -5.12392 |
| SCOM   | EGARCH(1,1)        |   0.000653 |      0.011765 | -7.4194  |
| SCOM   | 20-day rolling std |   0.000505 |      0.009198 | -7.08051 |
| ALL    | EGARCH(1,1)        |   0.001246 |      0.016242 | -6.97877 |
| ALL    | 20-day rolling std |   0.001241 |      0.012987 | -6.49719 |

## Chosen hyper-parameters
```json
{
  "xgb_base": {
    "max_depth": 3,
    "learning_rate": 0.02,
    "n_estimators": 500,
    "min_child_weight": 10
  },
  "xgb_hybrid": {
    "max_depth": 4,
    "learning_rate": 0.02,
    "n_estimators": 500,
    "min_child_weight": 50
  }
}
```

## Latest next-day forecasts (deployed model)
| code   | as_of_date   |   last_price |   forecast_return_pct |   forecast_price |   volatility_1d_pct |   range_95_low |   range_95_high |   base_model_price |   naive_price |
|:-------|:-------------|-------------:|----------------------:|-----------------:|--------------------:|---------------:|----------------:|-------------------:|--------------:|
| KCB    | 2026-06-30   |        78.75 |                 1.731 |           80.125 |               4.075 |         73.974 |          86.787 |             79.86  |         78.75 |
| KQ     | 2026-06-30   |         5.86 |                -0.229 |            5.847 |               2.746 |          5.54  |           6.17  |              5.849 |          5.86 |
| KEGN   | 2026-06-30   |         9.76 |                -0.233 |            9.737 |               2.401 |          9.29  |          10.206 |              9.73  |          9.76 |
| KNRE   | 2026-06-30   |         3.33 |                -0.122 |            3.326 |               2.219 |          3.184 |           3.474 |              3.326 |          3.33 |
| KPLC   | 2026-06-30   |        17.15 |                -0.139 |           17.126 |               2.035 |         16.457 |          17.823 |             17.12  |         17.15 |
| EQTY   | 2026-06-30   |        80    |                 1.205 |           80.97  |               1.98  |         77.888 |          84.174 |             80.79  |         80    |
| BRIT   | 2026-06-30   |        12.55 |                 0.189 |           12.574 |               1.869 |         12.122 |          13.043 |             12.568 |         12.55 |
| COOP   | 2026-06-30   |        34.3  |                -0.158 |           34.246 |               1.749 |         33.092 |          35.44  |             34.258 |         34.3  |
| ABSA   | 2026-06-30   |        32.05 |                 0.911 |           32.343 |               1.568 |         31.364 |          33.353 |             32.28  |         32.05 |
| SCOM   | 2026-06-30   |        34.05 |                 1.5   |           34.565 |               1.558 |         33.525 |          35.636 |             34.46  |         34.05 |
