#!/usr/bin/env python3
"""Sensitivity analysis for matched recession theory and sign-mixture variance.

This isolated post-processing analysis reads the frozen Stage-10, Stage-14,
and Stage-15 products and writes only to a versioned sensitivity directory. It
does not modify the numbered workflow stages.

The primary theory bridge estimates both moments from the same precipitation-
screened monotonic-decline transitions. For Delta q = -k q**b, it estimates
the direct conditional law Var(k | q) proportional to q**gamma and checks
m_decline = 2*b + gamma. Separately, it applies the law of total variance in
each starting-flow bin of the primary all-sign sample to separate within-sign
and between-sign contributions. The latter is a statistical decomposition,
not physical attribution.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from lib.common import (
    DEFAULT_CONFIG,
    atomic_json,
    atomic_target,
    load_config,
    output_root,
    sha256_file,
    stage_dir,
    utc_now,
)
from lib.stochastic import conditional_kramers_moyal, loglog_power_fit


ANALYSIS_VERSION = 2
DEFAULT_SEED = 20260909
DEFAULT_WATER_YEAR_BOOTSTRAPS = 199
DEFAULT_POPULATION_BOOTSTRAPS = 2000
FLOAT_RTOL = 1e-10
FLOAT_ATOL = 1e-12


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--water-year-bootstrap-replicates",
        type=int,
        default=DEFAULT_WATER_YEAR_BOOTSTRAPS,
    )
    parser.add_argument(
        "--population-bootstrap-replicates",
        type=int,
        default=DEFAULT_POPULATION_BOOTSTRAPS,
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory; defaults to the configured sensitivity-analysis root/"
            "theory_bridge_v2."
        ),
    )
    return parser.parse_args()


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    with atomic_target(path) as temporary:
        frame.to_csv(temporary, index=False)


def write_text_atomic(value: str, path: Path) -> None:
    with atomic_target(path) as temporary:
        temporary.write_text(value, encoding="utf-8")


def bool_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def clean_selected(frame: pd.DataFrame, flag: str) -> pd.DataFrame:
    selected = frame.loc[
        bool_mask(frame[flag]), ["date", "q_norm", "dq_norm_1d"]
    ].copy()
    selected["q_norm"] = pd.to_numeric(selected["q_norm"], errors="coerce")
    selected["dq_norm_1d"] = pd.to_numeric(
        selected["dq_norm_1d"], errors="coerce"
    )
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["q_norm", "dq_norm_1d"]
    )
    return selected[selected["q_norm"].gt(0)].sort_values("date")


def moment_fits(
    q: np.ndarray, dq: np.ndarray, cfg: dict
) -> tuple[pd.DataFrame, dict, dict]:
    settings = cfg["recession"]
    bins = conditional_kramers_moyal(
        q,
        dq,
        1,
        int(settings["state_bins"]),
        int(settings["minimum_bin_count"]),
    )
    variance = loglog_power_fit(
        bins["q_center"] if not bins.empty else np.array([]),
        bins["conditional_variance_rate"] if not bins.empty else np.array([]),
        int(settings["minimum_valid_bins_for_fit"]),
    )
    drift = loglog_power_fit(
        bins["q_center"] if not bins.empty else np.array([]),
        -bins["D1"] if not bins.empty else np.array([]),
        int(settings["minimum_valid_bins_for_fit"]),
    )
    return bins, variance, drift


def quantile_assignments(
    q: np.ndarray, state_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(q, np.linspace(0.0, 1.0, state_bins + 1)))
    if len(edges) < 4:
        return np.array([]), np.array([], dtype=int)
    return edges, np.searchsorted(edges[1:-1], q, side="right")


def rate_coefficient_bins(
    q: np.ndarray,
    dq: np.ndarray,
    b: float,
    state_bins: int,
    minimum_count: int,
) -> pd.DataFrame:
    edges, assignment = quantile_assignments(q, state_bins)
    if not len(edges):
        return pd.DataFrame()
    k = -dq / np.power(q, b)
    rows: list[dict] = []
    for state in range(len(edges) - 1):
        use = assignment == state
        n = int(use.sum())
        if n < minimum_count:
            continue
        q_bin, dq_bin, k_bin = q[use], dq[use], k[use]
        q_center = float(np.exp(np.mean(np.log(q_bin))))
        mean_dq = float(np.mean(dq_bin))
        variance_dq = float(np.mean((dq_bin - mean_dq) ** 2))
        mean_k = float(np.mean(k_bin))
        variance_k = float(np.mean((k_bin - mean_k) ** 2))
        center_scaled = float(variance_dq / np.power(q_center, 2.0 * b))
        rows.append(
            {
                "state_bin": state,
                "q_center": q_center,
                "q_min": float(np.min(q_bin)),
                "q_max": float(np.max(q_bin)),
                "n": n,
                "mean_delta_q": mean_dq,
                "variance_delta_q": variance_dq,
                "mean_k": mean_k,
                "variance_k_direct": variance_k,
                "variance_k_center_scaled": center_scaled,
                "direct_to_center_scaled_ratio": (
                    variance_k / center_scaled if center_scaled > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def sign_mixture_bins(
    q: np.ndarray,
    dq: np.ndarray,
    state_bins: int,
    minimum_count: int,
) -> pd.DataFrame:
    """Apply the population law of total variance within empirical q bins."""
    edges, assignment = quantile_assignments(q, state_bins)
    if not len(edges):
        return pd.DataFrame()
    rows: list[dict] = []
    for state in range(len(edges) - 1):
        use = assignment == state
        n = int(use.sum())
        if n < minimum_count:
            continue
        q_bin, values = q[use], dq[use]
        mean_all = float(np.mean(values))
        variance_all = float(np.mean((values - mean_all) ** 2))
        row: dict = {
            "state_bin": state,
            "q_center": float(np.exp(np.mean(np.log(q_bin)))),
            "q_min": float(np.min(q_bin)),
            "q_max": float(np.max(q_bin)),
            "n_all": n,
            "mean_all": mean_all,
            "variance_all": variance_all,
            "total_variance_rate": variance_all / 2.0,
        }
        masks = {
            "negative": values < 0,
            "zero": values == 0,
            "positive": values > 0,
        }
        within_total = 0.0
        between_total = 0.0
        for label, sign_use in masks.items():
            count = int(sign_use.sum())
            probability = count / n
            if count:
                signed = values[sign_use]
                signed_mean = float(np.mean(signed))
                signed_variance = float(np.mean((signed - signed_mean) ** 2))
            else:
                signed_mean = np.nan
                signed_variance = np.nan
            within = probability * (signed_variance if count else 0.0) / 2.0
            between = (
                probability * (signed_mean - mean_all) ** 2 / 2.0
                if count
                else 0.0
            )
            within_total += within
            between_total += between
            row.update(
                {
                    f"n_{label}": count,
                    f"p_{label}": probability,
                    f"mean_{label}": signed_mean,
                    f"variance_{label}": signed_variance,
                    f"conditional_{label}_variance_rate": (
                        signed_variance / 2.0 if count else np.nan
                    ),
                    f"within_{label}_contribution_rate": within,
                    f"between_{label}_contribution_rate": between,
                }
            )
        component_sum = within_total + between_total
        row["within_sign_contribution_rate"] = within_total
        row["between_sign_contribution_rate"] = between_total
        row["negative_within_contribution_rate"] = row[
            "within_negative_contribution_rate"
        ]
        row["positive_within_contribution_rate"] = row[
            "within_positive_contribution_rate"
        ]
        row["remainder_beyond_negative_within_rate"] = (
            component_sum - row["negative_within_contribution_rate"]
        )
        row["component_sum_variance_rate"] = component_sum
        row["closure_error_rate"] = component_sum - row["total_variance_rate"]
        denominator = row["total_variance_rate"]
        shares = {
            "negative_within_share": row["negative_within_contribution_rate"],
            "positive_within_share": row["positive_within_contribution_rate"],
            "between_sign_share": between_total,
            "remainder_beyond_negative_within_share": row[
                "remainder_beyond_negative_within_rate"
            ],
        }
        for label, value in shares.items():
            row[label] = value / denominator if denominator > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fitted_exponent(frame: pd.DataFrame, value: str, cfg: dict) -> dict:
    return loglog_power_fit(
        frame["q_center"] if not frame.empty else np.array([]),
        frame[value] if not frame.empty else np.array([]),
        int(cfg["recession"]["minimum_valid_bins_for_fit"]),
    )


def weighted_component_share(frame: pd.DataFrame, component: str) -> float:
    use = frame[["n_all", component, "total_variance_rate"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    denominator = float(np.sum(use["n_all"] * use["total_variance_rate"]))
    if denominator <= 0:
        return np.nan
    return float(np.sum(use["n_all"] * use[component]) / denominator)


def bootstrap_theory(
    selected: pd.DataFrame,
    cfg: dict,
    replicates: int,
    seed: int,
) -> dict:
    metrics = (
        "m_decline",
        "b",
        "delta_identity",
        "gamma_direct",
        "closure_direct",
    )
    result = {
        "water_year_bootstrap_requested": int(replicates),
        "water_year_bootstrap_valid": 0,
        "water_year_bootstrap_minimum_valid_required": (
            max(1, int(np.ceil(0.5 * replicates))) if replicates else 0
        ),
        "water_year_bootstrap_sufficient": False,
    }
    for metric in metrics:
        result[f"{metric}_wy_ci_low"] = np.nan
        result[f"{metric}_wy_ci_high"] = np.nan
    if replicates <= 0:
        return result
    dates = pd.to_datetime(selected["date"])
    years = (dates.dt.year + dates.dt.month.ge(10).astype(int)).to_numpy(int)
    unique_years = np.unique(years)
    if len(unique_years) < 5:
        return result
    q = selected["q_norm"].to_numpy(float)
    dq = selected["dq_norm_1d"].to_numpy(float)
    groups = [np.flatnonzero(years == year) for year in unique_years]
    settings = cfg["recession"]
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in metrics}
    for _ in range(replicates):
        picks = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in picks])
        q_boot, dq_boot = q[indices], dq[indices]
        _, variance, drift = moment_fits(q_boot, dq_boot, cfg)
        m_value, b_value = variance["exponent"], drift["exponent"]
        if not np.isfinite(m_value) or not np.isfinite(b_value):
            continue
        k_bins = rate_coefficient_bins(
            q_boot,
            dq_boot,
            b_value,
            int(settings["state_bins"]),
            int(settings["minimum_bin_count"]),
        )
        gamma = fitted_exponent(k_bins, "variance_k_direct", cfg)["exponent"]
        if not np.isfinite(gamma):
            continue
        delta = m_value - 2.0 * b_value
        samples["m_decline"].append(m_value)
        samples["b"].append(b_value)
        samples["delta_identity"].append(delta)
        samples["gamma_direct"].append(gamma)
        samples["closure_direct"].append(m_value - (2.0 * b_value + gamma))
    valid = len(samples["gamma_direct"])
    result["water_year_bootstrap_valid"] = int(valid)
    result["water_year_bootstrap_sufficient"] = bool(
        valid >= result["water_year_bootstrap_minimum_valid_required"]
    )
    if valid:
        for metric, values in samples.items():
            low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
            result[f"{metric}_wy_ci_low"] = float(low)
            result[f"{metric}_wy_ci_high"] = float(high)
    return result


def mixture_summary(mixture: pd.DataFrame, cfg: dict) -> dict:
    fit_columns = {
        "m_all_mixture": "total_variance_rate",
        "m_negative_conditional": "conditional_negative_variance_rate",
        "m_positive_conditional": "conditional_positive_variance_rate",
        "m_negative_within_contribution": "negative_within_contribution_rate",
        "m_positive_within_contribution": "positive_within_contribution_rate",
        "m_between_sign_contribution": "between_sign_contribution_rate",
        "m_remainder_beyond_negative_within": "remainder_beyond_negative_within_rate",
    }
    result: dict = {}
    for label, column in fit_columns.items():
        fit = fitted_exponent(mixture, column, cfg)
        result[label] = fit["exponent"]
        result[f"{label}_r2"] = fit["r2"]
        result[f"{label}_bins"] = fit["n_bins_fit"]
    for label, component in {
        "negative_within": "negative_within_contribution_rate",
        "positive_within": "positive_within_contribution_rate",
        "between_sign": "between_sign_contribution_rate",
        "remainder_beyond_negative_within": "remainder_beyond_negative_within_rate",
    }.items():
        result[f"weighted_{label}_share"] = weighted_component_share(
            mixture, component
        )
        result[f"median_bin_{label}_share"] = float(
            mixture[f"{label}_share"].median()
        )
    total = mixture["n_all"].sum()
    result["pooled_negative_transition_fraction"] = float(
        mixture["n_negative"].sum() / total
    )
    result["pooled_zero_transition_fraction"] = float(
        mixture["n_zero"].sum() / total
    )
    result["pooled_positive_transition_fraction"] = float(
        mixture["n_positive"].sum() / total
    )
    return result


def analyze_basin(
    task: tuple,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, list[dict]]:
    gage, source, frozen, cfg, wy_replicates, seed = task
    frame = pd.read_csv(
        source,
        usecols=[
            "date",
            "q_norm",
            "dq_norm_1d",
            "select_p_screened_monotone_1d",
            "select_p_screened_dry_state_1d",
        ],
        parse_dates=["date"],
        low_memory=False,
    )
    failures: list[dict] = []
    primary = clean_selected(frame, "select_p_screened_dry_state_1d")
    q_all = primary["q_norm"].to_numpy(float)
    dq_all = primary["dq_norm_1d"].to_numpy(float)
    _, all_variance, _ = moment_fits(q_all, dq_all, cfg)
    mixture = sign_mixture_bins(
        q_all,
        dq_all,
        int(cfg["recession"]["state_bins"]),
        int(cfg["recession"]["minimum_bin_count"]),
    )
    if mixture.empty or not np.isfinite(all_variance["exponent"]):
        raise ValueError("Primary all-sign variance fit or sign mixture unavailable")
    mixture.insert(0, "GAGE_ID", gage)
    row: dict = {
        "analysis_version": ANALYSIS_VERSION,
        "GAGE_ID": gage,
        "n_all_sign_transitions": len(primary),
        "m_all_frozen": frozen["m_all_frozen"],
        "m_all_recomputed": all_variance["exponent"],
        "m_all_recomputed_r2": all_variance["r2"],
        **mixture_summary(mixture, cfg),
        "theory_pair_available": False,
    }
    k_bins = pd.DataFrame()
    if np.isfinite(frozen["b_frozen"]) and np.isfinite(
        frozen["m_decline_frozen"]
    ):
        decline = clean_selected(frame, "select_p_screened_monotone_1d")
        q = decline["q_norm"].to_numpy(float)
        dq = decline["dq_norm_1d"].to_numpy(float)
        _, variance, drift = moment_fits(q, dq, cfg)
        m_decline, b = variance["exponent"], drift["exponent"]
        if not np.isfinite(m_decline) or not np.isfinite(b):
            failures.append(
                {
                    "GAGE_ID": gage,
                    "status": "theory_pair_recompute_failed",
                    "reason": "Frozen matched pair could not be recomputed",
                }
            )
        else:
            k_bins = rate_coefficient_bins(
                q,
                dq,
                b,
                int(cfg["recession"]["state_bins"]),
                int(cfg["recession"]["minimum_bin_count"]),
            )
            gamma_direct_fit = fitted_exponent(
                k_bins, "variance_k_direct", cfg
            )
            gamma_algebraic_fit = fitted_exponent(
                k_bins, "variance_k_center_scaled", cfg
            )
            gamma_direct = gamma_direct_fit["exponent"]
            gamma_algebraic = gamma_algebraic_fit["exponent"]
            delta_identity = m_decline - 2.0 * b
            k_bins.insert(0, "GAGE_ID", gage)
            k_bins["b_recomputed"] = b
            row.update(
                {
                    "theory_pair_available": True,
                    "n_decline_transitions": len(decline),
                    "m_decline_frozen": frozen["m_decline_frozen"],
                    "m_decline_recomputed": m_decline,
                    "m_decline_recomputed_r2": variance["r2"],
                    "b_frozen": frozen["b_frozen"],
                    "b_recomputed": b,
                    "b_recomputed_r2": drift["r2"],
                    "two_b_recomputed": 2.0 * b,
                    "delta_m_decline_minus_2b": delta_identity,
                    "delta_m_all_minus_2b": (
                        all_variance["exponent"] - 2.0 * b
                    ),
                    "selection_shift_m_decline_minus_all": (
                        m_decline - all_variance["exponent"]
                    ),
                    "gamma_direct": gamma_direct,
                    "gamma_direct_r2": gamma_direct_fit["r2"],
                    "gamma_direct_bins": gamma_direct_fit["n_bins_fit"],
                    "gamma_algebraic": gamma_algebraic,
                    "gamma_algebraic_r2": gamma_algebraic_fit["r2"],
                    "gamma_algebraic_bins": gamma_algebraic_fit["n_bins_fit"],
                    "algebraic_identity_error": (
                        gamma_algebraic - delta_identity
                    ),
                    "m_reconstructed_direct": 2.0 * b + gamma_direct,
                    "direct_reconstruction_error": (
                        m_decline - (2.0 * b + gamma_direct)
                    ),
                }
            )
            row.update(
                bootstrap_theory(
                    decline,
                    cfg,
                    wy_replicates,
                    seed + int(gage),
                )
            )
    return row, k_bins, mixture, failures


def lin_concordance(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return np.nan
    covariance = float(np.mean((x - np.mean(x)) * (y - np.mean(y))))
    denominator = float(
        np.var(x) + np.var(y) + (np.mean(x) - np.mean(y)) ** 2
    )
    return 2.0 * covariance / denominator if denominator > 0 else np.nan


def population_metrics(frame: pd.DataFrame) -> dict[str, tuple[float, int]]:
    theory = frame[frame["theory_pair_available"].fillna(False)].copy()
    metrics: dict[str, tuple[float, int]] = {}

    def median_metric(name: str, source: pd.DataFrame, column: str) -> None:
        values = pd.to_numeric(source[column], errors="coerce").dropna()
        metrics[name] = (
            float(values.median()) if len(values) else np.nan,
            len(values),
        )

    theory_medians = {
        "median_m_all_paired": "m_all_recomputed",
        "median_m_decline": "m_decline_recomputed",
        "median_two_b": "two_b_recomputed",
        "median_delta_m_decline_minus_2b": "delta_m_decline_minus_2b",
        "median_delta_m_all_minus_2b": "delta_m_all_minus_2b",
        "median_selection_shift_m_decline_minus_all": (
            "selection_shift_m_decline_minus_all"
        ),
        "median_gamma_direct": "gamma_direct",
        "median_gamma_algebraic": "gamma_algebraic",
        "median_direct_reconstruction_error": "direct_reconstruction_error",
    }
    for name, column in theory_medians.items():
        median_metric(name, theory, column)
    absolute = pd.to_numeric(
        theory["direct_reconstruction_error"], errors="coerce"
    ).abs().dropna()
    metrics["median_absolute_direct_reconstruction_error"] = (
        float(absolute.median()) if len(absolute) else np.nan,
        len(absolute),
    )

    x = pd.to_numeric(
        theory["m_decline_recomputed"], errors="coerce"
    ).to_numpy(float)
    y = pd.to_numeric(theory["two_b_recomputed"], errors="coerce").to_numpy(
        float
    )
    valid = np.isfinite(x) & np.isfinite(y)
    metrics["spearman_m_decline_vs_2b"] = (
        float(stats.spearmanr(x[valid], y[valid]).statistic),
        int(valid.sum()),
    )
    metrics["lin_concordance_m_decline_vs_2b"] = (
        lin_concordance(x, y),
        int(valid.sum()),
    )
    delta = pd.to_numeric(
        theory["delta_m_decline_minus_2b"], errors="coerce"
    ).dropna()
    metrics["fraction_abs_delta_decline_le_0p25"] = (
        float(delta.abs().le(0.25).mean()) if len(delta) else np.nan,
        len(delta),
    )
    comparison_differences = {
        "median_m_negative_conditional_minus_m_decline": (
            pd.to_numeric(theory["m_negative_conditional"], errors="coerce")
            - pd.to_numeric(theory["m_decline_recomputed"], errors="coerce")
        ),
        "median_m_remainder_minus_m_negative_conditional": (
            pd.to_numeric(
                theory["m_remainder_beyond_negative_within"], errors="coerce"
            )
            - pd.to_numeric(theory["m_negative_conditional"], errors="coerce")
        ),
    }
    for name, values in comparison_differences.items():
        values = values.dropna()
        metrics[name] = (
            float(values.median()) if len(values) else np.nan,
            len(values),
        )

    ci_columns = (
        (
            "fraction_wy_delta_ci_contains_zero",
            "delta_identity_wy_ci_low",
            "delta_identity_wy_ci_high",
        ),
        (
            "fraction_wy_gamma_direct_ci_contains_zero",
            "gamma_direct_wy_ci_low",
            "gamma_direct_wy_ci_high",
        ),
    )
    for name, low_name, high_name in ci_columns:
        low = pd.to_numeric(
            theory[low_name]
            if low_name in theory
            else pd.Series(np.nan, index=theory.index),
            errors="coerce",
        )
        high = pd.to_numeric(
            theory[high_name]
            if high_name in theory
            else pd.Series(np.nan, index=theory.index),
            errors="coerce",
        )
        bootstrap_supported = (
            theory["water_year_bootstrap_sufficient"].fillna(False).astype(bool)
            if "water_year_bootstrap_sufficient" in theory
            else pd.Series(False, index=theory.index)
        )
        available = low.notna() & high.notna() & bootstrap_supported
        metrics[name] = (
            float(((low[available] <= 0) & (high[available] >= 0)).mean())
            if available.any()
            else np.nan,
            int(available.sum()),
        )
        stem = name.removeprefix("fraction_wy_").removesuffix("_ci_contains_zero")
        metrics[f"fraction_wy_{stem}_ci_below_zero"] = (
            float((high[available] < 0).mean()) if available.any() else np.nan,
            int(available.sum()),
        )
        metrics[f"fraction_wy_{stem}_ci_above_zero"] = (
            float((low[available] > 0).mean()) if available.any() else np.nan,
            int(available.sum()),
        )

    mixture_medians = {
        "median_weighted_negative_within_share": (
            "weighted_negative_within_share"
        ),
        "median_weighted_positive_within_share": (
            "weighted_positive_within_share"
        ),
        "median_weighted_between_sign_share": "weighted_between_sign_share",
        "median_weighted_remainder_beyond_negative_within_share": (
            "weighted_remainder_beyond_negative_within_share"
        ),
        "median_bin_negative_within_share": "median_bin_negative_within_share",
        "median_bin_positive_within_share": "median_bin_positive_within_share",
        "median_bin_between_sign_share": "median_bin_between_sign_share",
        "median_pooled_positive_transition_fraction": (
            "pooled_positive_transition_fraction"
        ),
        "median_m_negative_conditional": "m_negative_conditional",
        "median_m_positive_conditional": "m_positive_conditional",
        "median_m_negative_within_contribution": (
            "m_negative_within_contribution"
        ),
        "median_m_positive_within_contribution": (
            "m_positive_within_contribution"
        ),
        "median_m_between_sign_contribution": "m_between_sign_contribution",
        "median_m_remainder_beyond_negative_within": (
            "m_remainder_beyond_negative_within"
        ),
    }
    for name, column in mixture_medians.items():
        median_metric(name, frame, column)
        median_metric(f"{name}_theory_paired", theory, column)
    return metrics


def population_bootstrap(
    basin: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if replicates <= 0:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    region_groups = {
        str(region): np.asarray(indices, dtype=int)
        for region, indices in basin.groupby(
            "AGGECOREGION", dropna=False
        ).indices.items()
    }
    regions = np.asarray(list(region_groups))
    for scheme in ("basin_iid", "macro_ecoregion_cluster"):
        for replicate in range(replicates):
            if scheme == "basin_iid":
                sampled = basin.iloc[
                    rng.integers(0, len(basin), size=len(basin))
                ]
            else:
                chosen = rng.choice(regions, size=len(regions), replace=True)
                indices = np.concatenate(
                    [region_groups[str(region)] for region in chosen]
                )
                sampled = basin.iloc[indices]
            for metric, (estimate, n_basin) in population_metrics(
                sampled
            ).items():
                rows.append(
                    {
                        "bootstrap_scheme": scheme,
                        "replicate": replicate + 1,
                        "metric": metric,
                        "estimate": estimate,
                        "n_basin": n_basin,
                    }
                )
    raw = pd.DataFrame(rows)
    intervals = (
        raw.groupby(["bootstrap_scheme", "metric"], sort=False)["estimate"]
        .agg(
            bootstrap_valid="count",
            ci_low_2p5=lambda values: values.quantile(0.025),
            bootstrap_median="median",
            ci_high_97p5=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )
    intervals["bootstrap_requested"] = replicates
    return raw, intervals


def methods_text(wy_replicates: int, population_replicates: int) -> str:
    return f"""# Matched recession-theory and sign-mixture sensitivity (version 2)

This isolated sensitivity analysis reads frozen Stage-10 transition tables and
Stage-14 estimates. It does not alter the completed numbered workflow.

## Matched theory bridge

Both the mean decline exponent `b` and conditional-variance exponent
`m_decline` are estimated from the same precipitation-screened monotonic-
decline transitions, using the original 20 equal-frequency starting-flow bins,
minimum 50 transitions per bin, and minimum six valid bins. The daily model is

`Delta q = -k q^b`.

For each transition, `k_i = -Delta q_i / q_i^b`. Within the same flow bins we
estimate `Var(k | q)` and fit `Var(k | q) = a_k q^gamma`. The theory identity is
`m_decline = 2b + gamma`; `gamma = 0` is the flow-independent coefficient-
variance reference. `gamma_direct` uses transition-level `k_i`. A second
`gamma_algebraic` scales the binned increment variance by each bin center and
therefore closes algebraically with `m_decline - 2b`. Their difference measures
finite bin-width effects, not a distinct physical mechanism.

Joint uncertainty is estimated with {wy_replicates} water-year block-bootstrap
replicates per matched basin: water years are sampled with replacement, bins
are reconstructed, and `m_decline`, `b`, and `gamma_direct` are re-estimated.
At least half of the requested replicates must be valid before a basin interval
is used in population summaries; the valid count and support flag are archived.
Population intervals use {population_replicates} iid-basin replicates and a
coarser nine-macro-ecoregion cluster bootstrap. The latter is a spatial
sensitivity, not a complete spatial covariance model.

## Exact sign-mixture decomposition

In every starting-flow bin of the primary all-sign dry-state sample, increments
are grouped as negative, zero, or positive. The population-variance identity is

`Var(Delta q | bin) = sum_s p_s Var(Delta q | bin,s)`
`                      + sum_s p_s [mu_s - mu]^2`.

Every term is divided by two days to match the reported variance-rate
convention. `closure_error_rate` stores the numerical difference between the
component sum and directly calculated total. Component exponents are
descriptive because a sum of power-law-like terms need not itself be one power
law. In particular, the remainder beyond the negative within-sign term must
not be labeled rainfall, operations, measurement error, or another physical
source without independent evidence. Also, the sustained monotonic-run sample
used for the theory bridge is not identical to merely selecting negative
increments from the primary all-sign sample.
"""


def maximum_abs_difference(left: pd.Series, right: pd.Series) -> float:
    difference = (
        pd.to_numeric(left, errors="coerce")
        - pd.to_numeric(right, errors="coerce")
    ).abs()
    return float(difference.max()) if difference.notna().any() else np.nan


def main() -> None:
    args = parse_arguments()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if (
        args.water_year_bootstrap_replicates < 0
        or args.population_bootstrap_replicates < 0
    ):
        raise ValueError("Bootstrap replicate counts cannot be negative")
    cfg = load_config(args.config)
    out = (
        Path(args.output).expanduser().resolve()
        if args.output
        else output_root(cfg) / "sensitivity_analyses" / "theory_bridge_v2"
    )
    receipt_path = out / "_SUCCESS.json"
    if receipt_path.exists() and not args.force:
        raise RuntimeError(
            f"Outputs already complete at {out}; use --force to replace them"
        )
    out.mkdir(parents=True, exist_ok=True)
    if args.force:
        receipt_path.unlink(missing_ok=True)

    required_receipts = [
        stage_dir(cfg, stage, create=False) / "_SUCCESS.json"
        for stage in (10, 14, 15)
    ]
    missing = [str(path) for path in required_receipts if not path.exists()]
    if missing:
        raise RuntimeError(f"Required frozen-stage receipts are missing: {missing}")

    stage14_path = (
        stage_dir(cfg, 14, create=False) / "recession_basin_method_summary.csv"
    )
    stage14 = pd.read_csv(
        stage14_path,
        dtype={"GAGE_ID": str},
        low_memory=False,
    )
    stage14["GAGE_ID"] = stage14["GAGE_ID"].astype(str).str.zfill(8)
    primary = stage14[
        stage14["method"].eq("p_screened_dry_state")
        & stage14["tau_days"].eq(1)
    ][["GAGE_ID", "variance_exponent"]].rename(
        columns={"variance_exponent": "m_all_frozen"}
    )
    primary["m_all_frozen"] = pd.to_numeric(
        primary["m_all_frozen"], errors="coerce"
    )
    primary = primary[primary["m_all_frozen"].notna()].copy()
    decline = stage14[
        stage14["method"].eq("p_screened_monotone")
        & stage14["tau_days"].eq(1)
    ][["GAGE_ID", "variance_exponent", "drift_exponent"]].rename(
        columns={
            "variance_exponent": "m_decline_frozen",
            "drift_exponent": "b_frozen",
        }
    )
    frozen = primary.merge(decline, on="GAGE_ID", how="left", validate="one_to_one")
    for column in ("m_decline_frozen", "b_frozen"):
        frozen[column] = pd.to_numeric(frozen[column], errors="coerce")
    frozen = frozen.sort_values("GAGE_ID").reset_index(drop=True)
    if args.limit:
        frozen = frozen.head(args.limit).copy()

    daily_dir = stage_dir(cfg, 10, create=False) / "daily"
    tasks = []
    for row in frozen.itertuples(index=False):
        gage = str(row.GAGE_ID).zfill(8)
        source = daily_dir / f"{gage}.csv.gz"
        if not source.exists():
            raise FileNotFoundError(source)
        tasks.append(
            (
                gage,
                str(source),
                {
                    "m_all_frozen": float(row.m_all_frozen),
                    "m_decline_frozen": float(row.m_decline_frozen),
                    "b_frozen": float(row.b_frozen),
                },
                cfg,
                args.water_year_bootstrap_replicates,
                DEFAULT_SEED,
            )
        )

    basin_rows: list[dict] = []
    k_parts: list[pd.DataFrame] = []
    mixture_parts: list[pd.DataFrame] = []
    failures: list[dict] = []
    unexpected = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_basin, task): task[0] for task in tasks}
        for number, future in enumerate(as_completed(futures), start=1):
            gage = futures[future]
            try:
                row, k_bins, mixture, basin_failures = future.result()
                basin_rows.append(row)
                if not k_bins.empty:
                    k_parts.append(k_bins)
                mixture_parts.append(mixture)
                failures.extend(basin_failures)
            except Exception as error:
                unexpected += 1
                failures.append(
                    {
                        "GAGE_ID": gage,
                        "status": "failed_unexpected",
                        "reason": repr(error),
                    }
                )
            if number % 100 == 0 or number == len(tasks):
                print(f"Completed {number}/{len(tasks)} basins", flush=True)
    if not basin_rows:
        raise RuntimeError("No basin completed the theory-bridge sensitivity")

    basin = pd.DataFrame(basin_rows).sort_values("GAGE_ID").reset_index(drop=True)
    k_bins = pd.concat(k_parts, ignore_index=True) if k_parts else pd.DataFrame()
    if not k_bins.empty:
        k_sort = [
            column for column in ("GAGE_ID", "state_bin") if column in k_bins.columns
        ]
        k_bins = k_bins.sort_values(k_sort).reset_index(drop=True)
    mixture = pd.concat(mixture_parts, ignore_index=True)
    mixture_sort = [
        column for column in ("GAGE_ID", "state_bin") if column in mixture.columns
    ]
    mixture = mixture.sort_values(mixture_sort).reset_index(drop=True)
    failures_frame = pd.DataFrame(
        failures, columns=["GAGE_ID", "status", "reason"]
    ).sort_values(["GAGE_ID", "status"]).reset_index(drop=True)

    attributes_path = (
        stage_dir(cfg, 15, create=False) / "continental_basin_results.csv"
    )
    attribute_columns = [
        "GAGE_ID",
        "quality_tier",
        "area_km2",
        "ecoregion",
        "AGGECOREGION",
        "LAT_GAGE",
        "LNG_GAGE",
        "STATE",
        "CLASS",
        "SNOW_PCT_PRECIP",
        "BFI_AVE",
    ]
    attributes = pd.read_csv(
        attributes_path,
        dtype={"GAGE_ID": str},
        usecols=attribute_columns,
        low_memory=False,
    )
    attributes["GAGE_ID"] = attributes["GAGE_ID"].astype(str).str.zfill(8)
    attributes = attributes.drop_duplicates("GAGE_ID")
    basin = basin.merge(attributes, on="GAGE_ID", how="left", validate="one_to_one")
    basin = basin.sort_values("GAGE_ID").reset_index(drop=True)
    k_bins = k_bins.sort_values(["GAGE_ID", "state_bin"]).reset_index(drop=True)
    mixture = mixture.sort_values(["GAGE_ID", "state_bin"]).reset_index(drop=True)

    point_rows = [
        {"metric": metric, "estimate": estimate, "n_basin": n_basin}
        for metric, (estimate, n_basin) in population_metrics(basin).items()
    ]
    point = pd.DataFrame(point_rows)
    bootstrap_raw, bootstrap_intervals = population_bootstrap(
        basin,
        args.population_bootstrap_replicates,
        DEFAULT_SEED + 50_000_000,
    )

    theory = basin[basin["theory_pair_available"].fillna(False)].copy()
    mixture_scale = mixture["total_variance_rate"].abs().clip(lower=np.finfo(float).tiny)
    mixture_relative_closure = (mixture["closure_error_rate"].abs() / mixture_scale)
    validation = {
        "analysis_version": ANALYSIS_VERSION,
        "basins_requested": len(tasks),
        "basins_completed": len(basin),
        "unexpected_failures": unexpected,
        "theory_pair_n": len(theory),
        "m_all_frozen_max_abs_difference": maximum_abs_difference(
            basin["m_all_frozen"], basin["m_all_recomputed"]
        ),
        "m_decline_frozen_max_abs_difference": maximum_abs_difference(
            theory["m_decline_frozen"], theory["m_decline_recomputed"]
        ),
        "b_frozen_max_abs_difference": maximum_abs_difference(
            theory["b_frozen"], theory["b_recomputed"]
        ),
        "algebraic_identity_max_abs_error": float(
            theory["algebraic_identity_error"].abs().max()
        ),
        "mixture_closure_max_abs_error_rate": float(
            mixture["closure_error_rate"].abs().max()
        ),
        "mixture_closure_max_relative_error": float(
            mixture_relative_closure.max()
        ),
        "direct_reconstruction_error_median": float(
            theory["direct_reconstruction_error"].median()
        ),
        "direct_reconstruction_error_q05": float(
            theory["direct_reconstruction_error"].quantile(0.05)
        ),
        "direct_reconstruction_error_q95": float(
            theory["direct_reconstruction_error"].quantile(0.95)
        ),
        "water_year_bootstrap_complete_n": int(
            theory["water_year_bootstrap_valid"].ge(
                args.water_year_bootstrap_replicates
            ).sum()
        )
        if args.water_year_bootstrap_replicates
        else 0,
        "water_year_bootstrap_any_valid_n": int(
            theory["water_year_bootstrap_valid"].gt(0).sum()
        )
        if args.water_year_bootstrap_replicates
        else 0,
        "water_year_bootstrap_sufficient_n": int(
            theory["water_year_bootstrap_sufficient"].fillna(False).sum()
        )
        if args.water_year_bootstrap_replicates
        else 0,
        "float_rtol": FLOAT_RTOL,
        "float_atol": FLOAT_ATOL,
    }
    validation["validation_pass"] = bool(
        unexpected == 0
        and len(basin) == len(tasks)
        and validation["m_all_frozen_max_abs_difference"] <= FLOAT_ATOL
        and validation["m_decline_frozen_max_abs_difference"] <= FLOAT_ATOL
        and validation["b_frozen_max_abs_difference"] <= FLOAT_ATOL
        and validation["algebraic_identity_max_abs_error"] <= FLOAT_ATOL
        and validation["mixture_closure_max_relative_error"] <= FLOAT_RTOL
    )

    outputs = {
        "basin": out / "basin_theory_bridge.csv",
        "k_bins": out / "rate_coefficient_variance_bins.csv",
        "mixture": out / "sign_mixture_variance_bins.csv",
        "point": out / "population_point_estimates.csv",
        "bootstrap_raw": out / "population_bootstrap_replicates.csv",
        "bootstrap_intervals": out / "population_bootstrap_intervals.csv",
        "failures": out / "failures.csv",
        "methods": out / "METHODS.md",
        "validation": out / "validation_summary.json",
    }
    write_csv_atomic(basin, outputs["basin"])
    write_csv_atomic(k_bins, outputs["k_bins"])
    write_csv_atomic(mixture, outputs["mixture"])
    write_csv_atomic(point, outputs["point"])
    write_csv_atomic(bootstrap_raw, outputs["bootstrap_raw"])
    write_csv_atomic(bootstrap_intervals, outputs["bootstrap_intervals"])
    write_csv_atomic(failures_frame, outputs["failures"])
    write_text_atomic(
        methods_text(
            args.water_year_bootstrap_replicates,
            args.population_bootstrap_replicates,
        ),
        outputs["methods"],
    )
    atomic_json(outputs["validation"], validation)

    if not validation["validation_pass"]:
        raise RuntimeError(
            f"Validation failed; outputs retained for diagnosis at {out}"
        )

    receipt = {
        "analysis": "Sensitivity analysis: matched theory bridge and sign mixture",
        "analysis_version": ANALYSIS_VERSION,
        "completed_utc": utc_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_file(Path(args.config).expanduser().resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "seed": DEFAULT_SEED,
        "water_year_bootstrap_replicates": args.water_year_bootstrap_replicates,
        "population_bootstrap_replicates": args.population_bootstrap_replicates,
        "limited_smoke_run": args.limit is not None,
        "limit": args.limit,
        "input_receipts": {
            str(path): sha256_file(path) for path in required_receipts
        },
        "input_tables": {
            str(stage14_path): sha256_file(stage14_path),
            str(attributes_path): sha256_file(attributes_path),
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in outputs.items()
        },
        "validation": validation,
        "interpretive_boundaries": [
            "The matched decline comparison is the primary recession-theory bridge.",
            "gamma_direct includes finite-bin-width effects; gamma_algebraic is the exact binned identity.",
            "The sign mixture is exact within bins but is not physical source attribution.",
            "The monotonic-run theory sample is not merely the negative subset of the all-sign sample.",
        ],
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps({"output": str(out), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
