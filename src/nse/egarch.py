"""EGARCH(1,1) volatility model, one per stock.

Parameters are estimated only on the training window. They are then held fixed and the
model is run forward over the whole series, so the volatility forecast for day t+1 uses
returns up to day t only (no look-ahead).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

log = logging.getLogger(__name__)
SCALE = 100.0  # arch is numerically happier with returns in percent


def _model(r: pd.Series, cfg: dict, dist: str):
    e = cfg["egarch"]
    return arch_model(r.fillna(0.0).to_numpy() * SCALE, mean="Constant", vol="EGARCH",
                      p=e["p"], o=e["o"], q=e["q"], dist=dist, rescale=False)


def fit_egarch(r: pd.Series, cfg: dict) -> dict:
    """Fit on the given (training) returns. Returns a JSON-safe dict."""
    for dist in (cfg["egarch"]["dist"], "normal"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = _model(r, cfg, dist).fit(disp="off", show_warning=False)
            if res.convergence_flag == 0 and np.all(np.isfinite(res.params)):
                return {"dist": dist, "params": res.params.tolist(),
                        "names": list(res.params.index), "loglik": float(res.loglikelihood),
                        "aic": float(res.aic), "nobs": int(res.nobs)}
        except Exception as exc:  # pragma: no cover
            log.warning("EGARCH fit with %s failed: %s", dist, exc)
    raise RuntimeError("EGARCH failed to converge")


def sigma_next(r: pd.Series, fitted: dict, cfg: dict) -> np.ndarray:
    """For every day t, the forecast standard deviation of the return on day t+1 (decimal)."""
    model = _model(r, cfg, fitted["dist"])
    fixed = model.fix(np.asarray(fitted["params"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = fixed.forecast(horizon=1, start=0, reindex=True)
    var = f.variance.to_numpy()[:, 0]
    return np.sqrt(var) / SCALE
