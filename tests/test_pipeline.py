import json

import pandas as pd


def test_pipeline_outputs(trained):
    m, r = trained / "models", trained / "reports"
    for f in ["xgb_base.json", "xgb_hybrid.json", "egarch_params.json", "metadata.json", "history.csv"]:
        assert (m / f).exists(), f
    for f in ["data_quality.csv", "test_metrics_overall.csv", "test_metrics_by_stock.csv",
              "test_volatility_metrics.csv", "test_diebold_mariano.csv", "RESULTS.md",
              "latest_forecasts.csv", "figures/test_prices.png"]:
        assert (r / f).exists(), f
    meta = json.loads((m / "metadata.json").read_text())
    assert set(meta["features_base"]) < set(meta["features_hybrid"])


def test_old_barclays_code_mapped(trained):
    q = pd.read_csv(trained / "reports" / "data_quality.csv")
    assert "ABSA" in set(q["code"]) and "BBK" not in set(q["code"])
    assert len(q) == 10


def test_test_period_is_2026_only(trained):
    p = pd.read_csv(trained / "reports" / "test_predictions.csv", parse_dates=["target_date"])
    assert p["target_date"].min() >= pd.Timestamp("2026-01-01")
