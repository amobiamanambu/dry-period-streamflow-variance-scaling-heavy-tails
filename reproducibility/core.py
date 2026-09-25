"""Small, exact analysis helpers used by the public demonstration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow.lib.stochastic import conditional_kramers_moyal, loglog_power_fit


def fit_primary_innovations(
    frame: pd.DataFrame, config: dict
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Fit the Stage-17 primary estimator to one basin's selected transitions."""
    selected = frame.loc[
        frame["select_p_screened_dry_state_1d"].fillna(False).astype(bool),
        ["date", "q_norm", "dq_norm_1d"],
    ].replace([np.inf, -np.inf], np.nan).dropna()
    selected = selected[selected["q_norm"].gt(0)].copy()

    setting = config["recession"]
    bins = conditional_kramers_moyal(
        selected["q_norm"], selected["dq_norm_1d"], 1,
        int(setting["state_bins"]), int(setting["minimum_bin_count"]),
    )
    fit = loglog_power_fit(
        bins["q_center"] if not bins.empty else np.array([]),
        bins["conditional_variance_rate"] if not bins.empty else np.array([]),
        int(setting["minimum_valid_bins_for_fit"]),
    )
    if not np.isfinite(fit["exponent"]) or not np.isfinite(fit["coefficient"]):
        raise ValueError("The example basin does not support the primary power-law fit")

    centers = bins["q_center"].to_numpy(float)
    conditional_means = bins["conditional_mean_increment"].to_numpy(float)
    order = np.argsort(centers)
    q = selected["q_norm"].to_numpy(float)
    dq = selected["dq_norm_1d"].to_numpy(float)
    mean = np.interp(
        np.log(q), np.log(centers[order]), conditional_means[order],
        left=conditional_means[order][0], right=conditional_means[order][-1],
    )
    sigma = np.sqrt(
        2.0 * float(fit["coefficient"]) * np.power(q, float(fit["exponent"]))
    )
    finite = np.isfinite(sigma) & (sigma > 0)
    retained = selected.iloc[np.flatnonzero(finite)].copy()
    raw = (dq[finite] - mean[finite]) / sigma[finite]
    finite_raw = np.isfinite(raw)
    retained = retained.iloc[np.flatnonzero(finite_raw)].copy()
    raw = raw[finite_raw]
    raw_mean = float(np.mean(raw))
    raw_sd = float(np.std(raw, ddof=1))
    z = (raw - raw_mean) / raw_sd
    retained["standardized_innovation"] = z

    threshold = float(config["distribution_test"]["normal_tail_z"])
    normal_one_sided = float(stats.norm.sf(threshold))
    positive = float(np.mean(z > threshold))
    negative = float(np.mean(z < -threshold))
    two_sided = positive + negative
    metrics = {
        "n_transitions": int(len(selected)),
        "n_standardized": int(len(z)),
        "variance_exponent": float(fit["exponent"]),
        "variance_coefficient": float(fit["coefficient"]),
        "variance_r2": float(fit["r2"]),
        "standardized_skew": float(stats.skew(z, bias=False)),
        "standardized_excess_kurtosis": float(stats.kurtosis(z, fisher=True, bias=False)),
        "positive_tail_fraction": positive,
        "negative_tail_fraction": negative,
        "two_sided_tail_fraction": two_sided,
        "positive_tail_multiple": positive / normal_one_sided,
        "negative_tail_multiple": negative / normal_one_sided,
        "two_sided_tail_multiple": two_sided / (2.0 * normal_one_sided),
    }
    return metrics, bins, retained


def tail_survival(z: np.ndarray, maximum: float = 6.0, points: int = 121) -> pd.DataFrame:
    """Return the unsmoothed empirical absolute-tail survival curve."""
    thresholds = np.linspace(0.0, maximum, points)
    values = np.asarray(z, dtype=float)
    values = values[np.isfinite(values)]
    return pd.DataFrame({
        "abs_z": thresholds,
        "empirical_exceedance": [float(np.mean(np.abs(values) > item)) for item in thresholds],
        "gaussian_exceedance": 2.0 * stats.norm.sf(thresholds),
    })
