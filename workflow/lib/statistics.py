"""Multiple testing, robust regression, resampling, and change-point helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p_values) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    p = values[valid]
    if not len(p):
        return adjusted
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty_like(q)
    restored[order] = np.clip(q, 0, 1)
    adjusted[valid] = restored
    return adjusted


def spearman_table(frame: pd.DataFrame, responses: list[str], predictors: list[str]) -> pd.DataFrame:
    rows = []
    for response in responses:
        for predictor in predictors:
            pair = frame[[response, predictor]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 20 or pair[response].nunique() < 3 or pair[predictor].nunique() < 3:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(pair[response], pair[predictor])
            rows.append({"response": response, "predictor": predictor, "n": len(pair), "rho": rho, "p_value": p})
    result = pd.DataFrame(rows)
    result["p_fdr"] = benjamini_hochberg(result["p_value"])
    return result


def robust_ols(frame: pd.DataFrame, response: str, predictors: list[str]) -> tuple[pd.DataFrame, dict]:
    import statsmodels.api as sm

    work = frame[[response, *predictors]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < max(30, 5 * len(predictors)):
        return pd.DataFrame(), {"n": len(work), "r2": np.nan, "rmse": np.nan}
    x = work[predictors].astype(float)
    sd = x.std(ddof=0).replace(0, np.nan)
    x = (x - x.mean()) / sd
    x = x.dropna(axis=1)
    model = sm.OLS(work.loc[x.index, response].astype(float), sm.add_constant(x)).fit(cov_type="HC3")
    table = pd.DataFrame({
        "term": model.params.index,
        "coefficient": model.params.values,
        "std_error_hc3": model.bse.values,
        "t": model.tvalues.values,
        "p_value": model.pvalues.values,
        "ci_low": model.conf_int().iloc[:, 0].values,
        "ci_high": model.conf_int().iloc[:, 1].values,
    })
    summary = {
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "adjusted_r2": float(model.rsquared_adj),
        "rmse": float(np.sqrt(np.mean(model.resid**2))),
    }
    return table, summary


def pettitt_test(values) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return {"change_index": np.nan, "statistic": np.nan, "p_value": np.nan}
    ranks = stats.rankdata(x)
    u = 2 * np.cumsum(ranks) - np.arange(1, n + 1) * (n + 1)
    index = int(np.argmax(np.abs(u[:-1])))
    statistic = float(abs(u[index]))
    p = min(1.0, 2.0 * np.exp((-6.0 * statistic**2) / (n**3 + n**2)))
    return {"change_index": index, "statistic": statistic, "p_value": float(p)}


def theil_sen_trend(years, values) -> dict:
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return {"sen_slope": np.nan, "sen_low": np.nan, "sen_high": np.nan, "kendall_tau": np.nan, "p_value": np.nan}
    slope, intercept, low, high = stats.theilslopes(y[valid], x[valid], alpha=0.95)
    tau, p = stats.kendalltau(x[valid], y[valid])
    return {
        "sen_slope": float(slope), "sen_low": float(low), "sen_high": float(high),
        "kendall_tau": float(tau), "p_value": float(p), "n": int(valid.sum()),
    }
