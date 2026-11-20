"""Metrics, Diebold-Mariano test and figures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def price_metrics(df: pd.DataFrame, pred_ret_col: str) -> dict:
    y = df["target_ret"].to_numpy()
    p = df[pred_ret_col].to_numpy()
    price_hat = df["price"].to_numpy() * np.exp(p)
    price = df["next_price"].to_numpy()
    nz = y != 0
    return {
        "n": int(len(df)),
        "ret_rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "ret_mae": float(np.mean(np.abs(y - p))),
        "price_rmse": float(np.sqrt(np.mean((price - price_hat) ** 2))),
        "price_mae": float(np.mean(np.abs(price - price_hat))),
        "price_mape_pct": float(np.mean(np.abs(price - price_hat) / price) * 100),
        # Direction only judged on days where the price actually moved.
        # A model that always predicts "no change" has no direction, so it gets NaN.
        "dir_acc": (float(np.mean(np.sign(p[nz]) == np.sign(y[nz])))
                    if nz.any() and np.any(p != 0) else np.nan),
        "days_moved": int(nz.sum()),
    }


def vol_metrics(df: pd.DataFrame, sigma_col: str) -> dict:
    """Compare a sigma forecast with the realised squared next-day return."""
    r2 = df["target_ret"].to_numpy() ** 2
    v = np.maximum(df[sigma_col].to_numpy(), 1e-6) ** 2
    return {
        "var_rmse": float(np.sqrt(np.mean((r2 - v) ** 2))),
        "abs_ret_mae": float(np.mean(np.abs(np.sqrt(r2) - np.sqrt(v)))),
        "qlike": float(np.mean(np.log(v) + r2 / v)),  # lower is better; robust to noisy proxy
    }


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM test on squared errors. Negative stat => model 1 more accurate."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    if n < 10 or np.var(d) == 0:
        return np.nan, np.nan
    dbar = d.mean()
    gamma = [np.cov(d[k:], d[:n - k])[0, 1] if k else np.var(d) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    stat = dbar / np.sqrt(var)
    adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # Harvey et al. correction
    stat *= adj
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


def plot_test_prices(test: pd.DataFrame, out_dir) -> None:
    codes = sorted(test["code"].unique())
    ncol = 2
    nrow = int(np.ceil(len(codes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.8 * nrow), squeeze=False)
    for ax, code in zip(axes.flat, codes):
        t = test[test["code"] == code]
        ax.plot(t["target_date"], t["next_price"], label="Actual", lw=1.4, color="black")
        ax.plot(t["target_date"], t["price"] * np.exp(t["pred_hybrid"]), label="Hybrid",
                lw=1, color="tab:green")
        lo = t["price"] * np.exp(-1.96 * t["egarch_sigma"])
        hi = t["price"] * np.exp(1.96 * t["egarch_sigma"])
        ax.fill_between(t["target_date"], lo, hi, color="tab:green", alpha=0.15, label="95% band")
        ax.set_title(code, fontsize=10)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(codes):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Held-out 2026: next-day price, hybrid forecast and EGARCH 95% band")
    fig.tight_layout()
    fig.savefig(out_dir / "test_prices.png", dpi=140)
    plt.close(fig)


def plot_volatility(test: pd.DataFrame, out_dir) -> None:
    codes = sorted(test["code"].unique())
    ncol = 2
    nrow = int(np.ceil(len(codes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.8 * nrow), squeeze=False)
    for ax, code in zip(axes.flat, codes):
        t = test[test["code"] == code]
        ax.bar(t["target_date"], t["target_ret"].abs(), color="lightgray", label="|actual return|")
        ax.plot(t["target_date"], t["egarch_sigma"], color="tab:red", lw=1.2, label="EGARCH sigma")
        ax.plot(t["target_date"], t["ret_std_20"], color="tab:blue", lw=1, ls="--", label="20-day std")
        ax.set_title(code, fontsize=10)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(codes):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Held-out 2026: volatility forecasts vs realised moves")
    fig.tight_layout()
    fig.savefig(out_dir / "test_volatility.png", dpi=140)
    plt.close(fig)


def plot_importance(model, features: list[str], out_path, title: str) -> pd.DataFrame:
    imp = model.get_booster().get_score(importance_type="gain")
    s = pd.Series({f: imp.get(f, 0.0) for f in features}).sort_values()
    s = s[~s.index.str.startswith("stock_")].tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    s.plot.barh(ax=ax, color="tab:green")
    ax.set_title(title)
    ax.set_xlabel("Gain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return s.sort_values(ascending=False).rename("gain").reset_index().rename(columns={"index": "feature"})
