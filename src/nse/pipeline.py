"""End-to-end pipeline:  ingest -> clean -> features -> tune -> test -> final fit -> artifacts.

Run:  python -m nse.pipeline            (everything)
      python -m nse.pipeline --skip-tuning   (reuse reports/best_params.json)
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import datetime, timezone

import arch
import numpy as np
import pandas as pd
import xgboost as xgb

from . import __version__
from .clean import clean
from .config import load_config
from .egarch import fit_egarch
from .evaluate import (diebold_mariano, plot_importance, plot_test_prices,
                       plot_volatility, price_metrics, vol_metrics)
from .features import BASE_FEATURES, VOL_FEATURES, build_features, stock_dummies
from .ingest import ingest
from .models import make_model, tune
from .predict import Forecaster

log = logging.getLogger("nse")
RAW_COLS = ["date", "code", "name", "price", "previous", "day_low", "day_high",
            "volume", "high_12m", "low_12m"]


def fit_all_egarch(df: pd.DataFrame, end: str | None, cfg: dict) -> dict:
    fits = {}
    for code, g in df.groupby("code", sort=True):
        r = g.loc[g["date"] <= pd.Timestamp(end), "ret"] if end else g["ret"]
        fits[code] = fit_egarch(r, cfg)
    return fits


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known target and enough history for rolling features."""
    return df[df["target_ret"].notna() & df["ret_std_20"].notna() & df["egarch_sigma"].notna()]


def egarch_table(fits: dict, stage: str) -> pd.DataFrame:
    rows = []
    for code, f in fits.items():
        row = {"stage": stage, "code": code, "dist": f["dist"], "aic": f["aic"], "nobs": f["nobs"]}
        row.update(dict(zip(f["names"], f["params"])))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-tuning", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.time()
    cfg = load_config(args.config)
    P = cfg["paths"]
    for k in ("processed_dir", "models_dir", "reports_dir"):
        P[k].mkdir(parents=True, exist_ok=True)
    fig_dir = P["reports_dir"] / "figures"
    fig_dir.mkdir(exist_ok=True)
    S = cfg["split"]
    np.random.seed(cfg["seed"])

    # 1-2. Ingest and clean ---------------------------------------------------------
    log.info("STEP 1/6  Ingest")
    raw = ingest(P["raw_dir"])
    log.info("STEP 2/6  Clean")
    data, quality = clean(raw, cfg)
    data.to_csv(P["processed_dir"] / "clean.csv", index=False)
    quality.to_csv(P["reports_dir"] / "data_quality.csv", index=False)

    codes = sorted(data["code"].unique())
    f_base = BASE_FEATURES + stock_dummies(cfg["stocks"])
    f_hyb = BASE_FEATURES + VOL_FEATURES + stock_dummies(cfg["stocks"])

    # 3. Tuning stage: train <= train_end, validate on (train_end, val_end] ----------
    best_path = P["reports_dir"] / "best_params.json"
    if args.skip_tuning and best_path.exists():
        log.info("STEP 3/6  Tuning skipped - using %s", best_path.name)
        best = json.loads(best_path.read_text())
    else:
        log.info("STEP 3/6  Tune on train <= %s, validate to %s", S["train_end"], S["val_end"])
        fits_a = fit_all_egarch(data, S["train_end"], cfg)
        feat_a = usable(build_features(data, fits_a, cfg))
        tr = feat_a[feat_a["target_date"] <= S["train_end"]]
        va = feat_a[(feat_a["target_date"] > S["train_end"]) & (feat_a["target_date"] <= S["val_end"])]
        log.info("          train rows %d | validation rows %d", len(tr), len(va))
        best_b, res_b = tune(tr, va, f_base, cfg, "xgb_base")
        best_h, res_h = tune(tr, va, f_hyb, cfg, "xgb_hybrid")
        pd.concat([res_b, res_h]).to_csv(P["reports_dir"] / "tuning_results.csv", index=False)
        best = {"xgb_base": best_b, "xgb_hybrid": best_h}
        best_path.write_text(json.dumps(best, indent=2))

    # 4. Final test: refit on <= val_end, evaluate once on 2026 ----------------------
    log.info("STEP 4/6  Refit on data <= %s and test on %s onwards", S["val_end"], S["test_start"])
    fits_b = fit_all_egarch(data, S["val_end"], cfg)
    feat_b = usable(build_features(data, fits_b, cfg))
    tr = feat_b[feat_b["target_date"] <= S["val_end"]]
    te = feat_b[feat_b["target_date"] >= S["test_start"]].copy()
    if te.empty:
        raise RuntimeError(f"No test rows on/after {S['test_start']} - check your 2026 file.")
    m_base = make_model(best["xgb_base"], cfg).fit(tr[f_base], tr["target_ret"])
    m_hyb = make_model(best["xgb_hybrid"], cfg).fit(tr[f_hyb], tr["target_ret"])
    te["pred_naive"] = 0.0
    te["pred_base"] = m_base.predict(te[f_base])
    te["pred_hybrid"] = m_hyb.predict(te[f_hyb])
    log.info("          train rows %d | test rows %d (%s -> %s)", len(tr), len(te),
             te["target_date"].min().date(), te["target_date"].max().date())

    names = {"naive": "Last price (naive)", "base": "XGBoost (no volatility)",
             "hybrid": "EGARCH + XGBoost (hybrid)"}
    overall, by_stock, dm_rows, vol_rows = [], [], [], []
    for key, label in names.items():
        overall.append({"model": label, **price_metrics(te, f"pred_{key}")})
    for code, t in te.groupby("code"):
        for key, label in names.items():
            by_stock.append({"code": code, "model": label, **price_metrics(t, f"pred_{key}")})
        for other in ("base", "naive"):
            s, p = diebold_mariano(t["target_ret"] - t["pred_hybrid"], t["target_ret"] - t[f"pred_{other}"])
            dm_rows.append({"code": code, "hybrid_vs": names[other], "dm_stat": s, "p_value": p})
        for col, label in (("egarch_sigma", "EGARCH(1,1)"), ("ret_std_20", "20-day rolling std")):
            vol_rows.append({"code": code, "model": label, **vol_metrics(t, col)})
    for other in ("base", "naive"):
        s, p = diebold_mariano(te["target_ret"] - te["pred_hybrid"], te["target_ret"] - te[f"pred_{other}"])
        dm_rows.append({"code": "ALL", "hybrid_vs": names[other], "dm_stat": s, "p_value": p})
    for col, label in (("egarch_sigma", "EGARCH(1,1)"), ("ret_std_20", "20-day rolling std")):
        vol_rows.append({"code": "ALL", "model": label, **vol_metrics(te, col)})

    R = P["reports_dir"]
    overall = pd.DataFrame(overall)
    overall.to_csv(R / "test_metrics_overall.csv", index=False)
    pd.DataFrame(by_stock).to_csv(R / "test_metrics_by_stock.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(R / "test_diebold_mariano.csv", index=False)
    vol = pd.DataFrame(vol_rows)
    vol.to_csv(R / "test_volatility_metrics.csv", index=False)
    pd.concat([egarch_table(fits_b, "test_stage")]).to_csv(R / "egarch_params.csv", index=False)
    te[["code", "date", "target_date", "price", "next_price", "target_ret", "pred_naive",
        "pred_base", "pred_hybrid", "egarch_sigma", "ret_std_20"]].to_csv(
        R / "test_predictions.csv", index=False)
    plot_test_prices(te, fig_dir)
    plot_volatility(te, fig_dir)
    plot_importance(m_hyb, f_hyb, fig_dir / "importance_hybrid.png",
                    "Hybrid model - feature importance").to_csv(R / "importance_hybrid.csv", index=False)
    plot_importance(m_base, f_base, fig_dir / "importance_base.png", "Base model - feature importance")
    log.info("\n%s", overall.round(5).to_string(index=False))

    # 5. Production fit on all data ---------------------------------------------------
    log.info("STEP 5/6  Final fit on all data for deployment")
    fits_c = fit_all_egarch(data, None, cfg)
    feat_c = usable(build_features(data, fits_c, cfg))
    final_base = make_model(best["xgb_base"], cfg).fit(feat_c[f_base], feat_c["target_ret"])
    final_hyb = make_model(best["xgb_hybrid"], cfg).fit(feat_c[f_hyb], feat_c["target_ret"])
    M = P["models_dir"]
    final_base.save_model(M / "xgb_base.json")
    final_hyb.save_model(M / "xgb_hybrid.json")
    (M / "egarch_params.json").write_text(json.dumps(fits_c, indent=2))
    hist = (data.sort_values("date").groupby("code").tail(cfg["serving"]["history_rows"]))
    hist[RAW_COLS].to_csv(M / "history.csv", index=False)
    cfg_json = json.loads(json.dumps(cfg, default=str))
    meta = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_range": [str(data["date"].min().date()), str(data["date"].max().date())],
        "train_rows": int(len(feat_c)),
        "features_base": f_base,
        "features_hybrid": f_hyb,
        "best_params": best,
        "test_metrics": overall.to_dict(orient="records"),
        "versions": {"python": platform.python_version(), "xgboost": xgb.__version__,
                     "arch": arch.__version__, "pandas": pd.__version__},
        "config": cfg_json,
    }
    (M / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    # 6. Latest forecasts + summary -----------------------------------------------------
    log.info("STEP 6/6  Latest forecasts and summary")
    fc = pd.DataFrame(Forecaster(M).forecast_all())
    fc.to_csv(R / "latest_forecasts.csv", index=False)
    write_summary(R, quality, overall, pd.DataFrame(by_stock), pd.DataFrame(dm_rows), vol, fc, best, S)
    log.info("Done in %.1f s. Artifacts in %s, reports in %s", time.time() - t0, M, R)


def _md(df: pd.DataFrame, digits: int = 4) -> str:
    return df.round(digits).to_markdown(index=False)


def write_summary(R, quality, overall, by_stock, dm, vol, fc, best, S) -> None:
    hyb = by_stock[by_stock["model"].str.startswith("EGARCH")].set_index("code")
    base = by_stock[by_stock["model"].str.startswith("XGBoost")].set_index("code")
    naive = by_stock[by_stock["model"].str.startswith("Last")].set_index("code")
    cmp = pd.DataFrame({
        "price_mape_hybrid": hyb["price_mape_pct"],
        "price_mape_base": base["price_mape_pct"],
        "price_mape_naive": naive["price_mape_pct"],
        "dir_acc_hybrid": hyb["dir_acc"],
        "dir_acc_base": base["dir_acc"],
    }).reset_index()
    text = f"""# Results summary (auto-generated)

Tuning: train to {S['train_end']}, validate to {S['val_end']}. Final test: {S['test_start']} onwards (used once).

## Data quality
{_md(quality.drop(columns=['name']), 3)}

## Held-out test - all stocks
{_md(overall, 5)}

## Held-out test - by stock
{_md(cmp, 4)}

## Diebold-Mariano (squared return errors; negative = hybrid better)
{_md(dm, 4)}

## Volatility forecasts (lower is better)
{_md(vol, 6)}

## Chosen hyper-parameters
```json
{json.dumps(best, indent=2)}
```

## Latest next-day forecasts (deployed model)
{_md(fc.drop(columns=['name']), 3)}
"""
    (R / "RESULTS.md").write_text(text)


if __name__ == "__main__":
    main()
