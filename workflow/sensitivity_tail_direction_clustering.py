#!/usr/bin/env python3
"""Sensitivity analysis for signed tails, within-spell clustering, and zero-flow context.

This is an isolated, versioned post-processing analysis.  It reads the frozen
Stage-10 daily transition files and reproduces the Stage-17 primary
standardization before adding diagnostics that were not archived in Stage 17.
It never writes to Stages 01--20.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
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


ANALYSIS_VERSION = 1
PROFILE_THRESHOLDS = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
MINIMUM_CLUSTERING_PAIRS = 100
MINIMUM_CORRELATION_PAIRS = 3
FLOAT_RTOL = 1e-10
FLOAT_ATOL = 1e-12


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output directory; defaults to the configured sensitivity-analysis directory.",
    )
    return parser.parse_args()


def write_csv_atomic(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    with atomic_target(path) as temporary:
        frame.to_csv(temporary, index=False, **kwargs)


def write_text_atomic(text: str, path: Path) -> None:
    with atomic_target(path) as temporary:
        temporary.write_text(text, encoding="utf-8")


def exponent_fit(selected: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Exact copy of the Stage-17 conditional-variance fitting operations."""
    setting = cfg["recession"]
    km = conditional_kramers_moyal(
        selected["q_norm"],
        selected["dq_norm_1d"],
        1,
        setting["state_bins"],
        setting["minimum_bin_count"],
    )
    fit = loglog_power_fit(
        km["q_center"] if not km.empty else np.array([]),
        km["conditional_variance_rate"] if not km.empty else np.array([]),
        setting["minimum_valid_bins_for_fit"],
    )
    return km, fit


def interpolate_conditional_mean(q: np.ndarray, km: pd.DataFrame) -> np.ndarray:
    """Exact copy of the Stage-17 log-flow interpolation."""
    centers = km["q_center"].to_numpy(float)
    means = km["conditional_mean_increment"].to_numpy(float)
    order = np.argsort(centers)
    return np.interp(
        np.log(q),
        np.log(centers[order]),
        means[order],
        left=means[order][0],
        right=means[order][-1],
    )


def standardize_like_stage17(
    selected: pd.DataFrame, cfg: dict
) -> tuple[dict, pd.DataFrame]:
    """Reproduce Stage-17 fitting, centering, and unit-variance rescaling."""
    km, fit = exponent_fit(selected, cfg)
    if not np.isfinite(fit["exponent"]) or not np.isfinite(fit["coefficient"]):
        raise ValueError("Conditional variance power fit unavailable")

    q = selected["q_norm"].to_numpy(float)
    dq = selected["dq_norm_1d"].to_numpy(float)
    conditional_mean = interpolate_conditional_mean(q, km)
    sigma = np.sqrt(2.0 * fit["coefficient"] * np.power(q, fit["exponent"]))
    valid = np.isfinite(sigma) & (sigma > 0)
    retained = selected.iloc[np.flatnonzero(valid)].copy()
    q = q[valid]
    raw_z = (dq[valid] - conditional_mean[valid]) / sigma[valid]
    finite = np.isfinite(raw_z)
    retained = retained.iloc[np.flatnonzero(finite)].copy()
    raw_z = raw_z[finite]
    raw_mean = float(np.mean(raw_z))
    raw_sd = float(np.std(raw_z, ddof=1))
    if not np.isfinite(raw_sd) or raw_sd <= 0:
        raise ValueError("Power-scaled residual variance unavailable")
    z = (raw_z - raw_mean) / raw_sd
    retained["standardized_innovation"] = z
    return {
        "n_standardized": len(z),
        "variance_exponent": float(fit["exponent"]),
        "variance_coefficient": float(fit["coefficient"]),
        "variance_r2": float(fit["r2"]),
        "power_scaled_mean_before_global_centering": raw_mean,
        "power_scaled_sd_before_global_rescaling": raw_sd,
        "standardized_mean": float(np.mean(z)),
        "standardized_sd": float(np.std(z, ddof=1)),
        "standardized_skew": float(stats.skew(z, bias=False)),
        "standardized_excess_kurtosis": float(
            stats.kurtosis(z, fisher=True, bias=False)
        ),
    }, retained


def jeffreys_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    return (
        float(stats.beta.ppf(0.025, successes + 0.5, trials - successes + 0.5)),
        float(stats.beta.ppf(0.975, successes + 0.5, trials - successes + 0.5)),
    )


def fisher_interval(r: float, n: int) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r):
        return np.nan, np.nan
    if abs(r) >= 1:
        return float(r), float(r)
    transformed = np.arctanh(r)
    width = stats.norm.ppf(0.975) / math.sqrt(n - 3)
    return float(np.tanh(transformed - width)), float(np.tanh(transformed + width))


def correlation_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = len(x)
    pearson = np.nan
    spearman = np.nan
    if (
        n >= MINIMUM_CORRELATION_PAIRS
        and np.ptp(x) > 0
        and np.ptp(y) > 0
    ):
        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman = float(stats.spearmanr(x, y).statistic)
    ci_low, ci_high = fisher_interval(pearson, n)
    return {
        "n": n,
        "pearson": pearson,
        "pearson_ci_low": ci_low,
        "pearson_ci_high": ci_high,
        "spearman": spearman,
    }


def directional_metrics(
    retained: pd.DataFrame,
    fit_metrics: dict,
    cfg: dict,
    variant: str,
    n_input: int,
    low_flow_cutoff_cfs: float = np.nan,
) -> tuple[dict, pd.DataFrame]:
    """Calculate signed-tail and strictly within-consecutive-run diagnostics."""
    work = retained.sort_values("date").reset_index(drop=True)
    z = work["standardized_innovation"].to_numpy(float)
    dates = pd.to_datetime(work["date"])
    n = len(z)
    threshold = float(cfg["distribution_test"]["normal_tail_z"])
    positive_count = int(np.sum(z > threshold))
    negative_count = int(np.sum(z < -threshold))
    tail_count = positive_count + negative_count
    positive_fraction = positive_count / n
    negative_fraction = negative_count / n
    two_sided_fraction = tail_count / n
    normal_one_sided = float(stats.norm.sf(threshold))
    positive_ci = jeffreys_interval(positive_count, n)
    negative_ci = jeffreys_interval(negative_count, n)

    consecutive = dates.diff().dt.days.eq(1).to_numpy(copy=True)
    pair_indices = np.flatnonzero(consecutive)
    previous_z = z[pair_indices - 1]
    current_z = z[pair_indices]
    abs_corr = correlation_metrics(np.abs(previous_z), np.abs(current_z))
    square_corr = correlation_metrics(previous_z ** 2, current_z ** 2)

    new_run = ~consecutive
    if len(new_run):
        new_run[0] = True
    run_ids = np.cumsum(new_run)
    run_lengths = pd.Series(run_ids).value_counts().to_numpy(int) if n else np.array([])
    support_class = (
        "adequate"
        if len(pair_indices) >= MINIMUM_CLUSTERING_PAIRS
        else "low"
        if len(pair_indices) >= 30
        else "insufficient"
    )

    row = {
        "analysis_version": ANALYSIS_VERSION,
        "screen_variant": variant,
        "n_input": int(n_input),
        **fit_metrics,
        "tail_threshold_abs_z": threshold,
        "positive_tail_count": positive_count,
        "negative_tail_count": negative_count,
        "two_sided_tail_count": tail_count,
        "positive_tail_fraction": positive_fraction,
        "positive_tail_fraction_jeffreys_low": positive_ci[0],
        "positive_tail_fraction_jeffreys_high": positive_ci[1],
        "negative_tail_fraction": negative_fraction,
        "negative_tail_fraction_jeffreys_low": negative_ci[0],
        "negative_tail_fraction_jeffreys_high": negative_ci[1],
        "two_sided_tail_fraction": two_sided_fraction,
        "normal_one_sided_tail_reference": normal_one_sided,
        "normal_two_sided_tail_reference": 2.0 * normal_one_sided,
        "positive_tail_multiple_of_normal": positive_fraction / normal_one_sided,
        "negative_tail_multiple_of_normal": negative_fraction / normal_one_sided,
        "two_sided_tail_multiple_of_normal": two_sided_fraction / (2.0 * normal_one_sided),
        "signed_tail_fraction_difference": positive_fraction - negative_fraction,
        "raw_positive_share_of_extreme_tails": (
            positive_count / tail_count if tail_count else np.nan
        ),
        "jeffreys_positive_share_of_extreme_tails": (
            (positive_count + 0.5) / (tail_count + 1.0)
        ),
        "jeffreys_tail_log_count_ratio": float(
            np.log((positive_count + 0.5) / (negative_count + 0.5))
        ),
        "jeffreys_tail_asymmetry_index": (
            (positive_count - negative_count) / (tail_count + 1.0)
        ),
        "no_abs_z_gt_threshold": bool(tail_count == 0),
        "n_consecutive_within_spell_pairs": int(len(pair_indices)),
        "fraction_possible_pairs_consecutive": (
            len(pair_indices) / (n - 1) if n > 1 else np.nan
        ),
        "retained_screened_spell_count": int(len(run_lengths)),
        "median_retained_transitions_per_spell": (
            float(np.median(run_lengths)) if len(run_lengths) else np.nan
        ),
        "maximum_retained_transitions_per_spell": (
            int(np.max(run_lengths)) if len(run_lengths) else 0
        ),
        "clustering_minimum_pairs": MINIMUM_CLUSTERING_PAIRS,
        "clustering_low_support": bool(len(pair_indices) < MINIMUM_CLUSTERING_PAIRS),
        "clustering_support_class": support_class,
        "abs_z_lag1_pearson": abs_corr["pearson"],
        "abs_z_lag1_pearson_ci_low": abs_corr["pearson_ci_low"],
        "abs_z_lag1_pearson_ci_high": abs_corr["pearson_ci_high"],
        "abs_z_lag1_spearman": abs_corr["spearman"],
        "z2_lag1_pearson": square_corr["pearson"],
        "z2_lag1_pearson_ci_low": square_corr["pearson_ci_low"],
        "z2_lag1_pearson_ci_high": square_corr["pearson_ci_high"],
        "z2_lag1_spearman": square_corr["spearman"],
        "conservative_low_flow_cutoff_cfs": low_flow_cutoff_cfs,
    }

    profile_rows = []
    for profile_threshold in PROFILE_THRESHOLDS:
        pos = int(np.sum(z > profile_threshold))
        neg = int(np.sum(z < -profile_threshold))
        gaussian = float(stats.norm.sf(profile_threshold))
        profile_rows.append({
            "screen_variant": variant,
            "threshold_abs_z": profile_threshold,
            "n_standardized": n,
            "positive_tail_count": pos,
            "negative_tail_count": neg,
            "positive_tail_fraction": pos / n,
            "negative_tail_fraction": neg / n,
            "two_sided_tail_fraction": (pos + neg) / n,
            "normal_one_sided_tail_reference": gaussian,
            "normal_two_sided_tail_reference": 2.0 * gaussian,
            "positive_tail_multiple_of_normal": pos / n / gaussian,
            "negative_tail_multiple_of_normal": neg / n / gaussian,
        })
    return row, pd.DataFrame(profile_rows)


def zero_flow_context(frame: pd.DataFrame) -> dict:
    q = pd.to_numeric(frame["q_cfs"], errors="coerce")
    q_next = q.shift(-1)
    dates = pd.to_datetime(frame["date"])
    observed = q.notna() & np.isfinite(q)
    next_observed = q_next.notna() & np.isfinite(q_next)
    zero = observed & q.eq(0.0)
    positive = observed & q.gt(0.0)
    negative = observed & q.lt(0.0)
    consecutive_next = (dates.shift(-1) - dates).dt.days.eq(1)
    all_finite_endpoints = consecutive_next & observed & next_observed
    all_origin_zero = all_finite_endpoints & q.eq(0.0)
    all_destination_zero = all_finite_endpoints & q_next.eq(0.0)
    all_touches_zero = all_origin_zero | all_destination_zero
    all_zero_to_zero = all_finite_endpoints & q.eq(0.0) & q_next.eq(0.0)
    all_positive_to_zero = all_finite_endpoints & q.gt(0.0) & q_next.eq(0.0)
    all_zero_to_positive = all_finite_endpoints & q.eq(0.0) & q_next.gt(0.0)
    all_positive_to_positive = all_finite_endpoints & q.gt(0.0) & q_next.gt(0.0)
    flag = frame["select_p_screened_dry_state_1d"].fillna(False).astype(bool)
    q_norm = pd.to_numeric(frame["q_norm"], errors="coerce")
    dq = pd.to_numeric(frame["dq_norm_1d"], errors="coerce")
    complete = flag & q_norm.notna() & np.isfinite(q_norm) & dq.notna() & np.isfinite(dq)
    finite_endpoints = complete & q.notna() & q_next.notna()
    origin_zero = finite_endpoints & q.eq(0.0)
    destination_zero = finite_endpoints & q_next.eq(0.0)
    touches_zero = origin_zero | destination_zero
    zero_to_zero = finite_endpoints & q.eq(0.0) & q_next.eq(0.0)
    positive_to_zero = finite_endpoints & q.gt(0.0) & q_next.eq(0.0)
    zero_to_positive = finite_endpoints & q.eq(0.0) & q_next.gt(0.0)

    selected_dates = pd.to_datetime(frame.loc[flag, "date"]).sort_values().reset_index(drop=True)
    consecutive = selected_dates.diff().dt.days.eq(1).to_numpy(copy=True)
    if len(consecutive):
        consecutive[0] = False
    run_count = int((~consecutive).sum()) if len(consecutive) else 0

    observed_n = int(observed.sum())
    all_finite_endpoint_n = int(all_finite_endpoints.sum())
    finite_endpoint_n = int(finite_endpoints.sum())
    selected_n = int(flag.sum())
    return {
        "q_observed_n": observed_n,
        "q_exact_zero_n": int(zero.sum()),
        "q_positive_n": int(positive.sum()),
        "q_negative_n": int(negative.sum()),
        "q_exact_zero_fraction_of_observed": (
            float(zero.sum() / observed_n) if observed_n else np.nan
        ),
        "all_consecutive_finite_discharge_transition_n": all_finite_endpoint_n,
        "all_origin_exact_zero_transition_n": int(all_origin_zero.sum()),
        "all_destination_exact_zero_transition_n": int(all_destination_zero.sum()),
        "all_transition_touching_exact_zero_n": int(all_touches_zero.sum()),
        "all_transition_touching_exact_zero_fraction": (
            float(all_touches_zero.sum() / all_finite_endpoint_n)
            if all_finite_endpoint_n else np.nan
        ),
        "all_zero_to_zero_transition_n": int(all_zero_to_zero.sum()),
        "all_positive_to_zero_transition_n": int(all_positive_to_zero.sum()),
        "all_zero_to_positive_transition_n": int(all_zero_to_positive.sum()),
        "all_positive_to_positive_transition_n": int(all_positive_to_positive.sum()),
        "primary_flagged_transition_n": selected_n,
        "primary_flagged_screened_spell_count": run_count,
        "primary_complete_transition_n": int(complete.sum()),
        "primary_positive_origin_transition_n": int((complete & q_norm.gt(0)).sum()),
        "primary_nonpositive_origin_transition_n": int((complete & q_norm.le(0)).sum()),
        "primary_finite_discharge_endpoint_transition_n": finite_endpoint_n,
        "primary_origin_exact_zero_n": int(origin_zero.sum()),
        "primary_destination_exact_zero_n": int(destination_zero.sum()),
        "primary_transition_touching_exact_zero_n": int(touches_zero.sum()),
        "primary_transition_touching_exact_zero_fraction": (
            float(touches_zero.sum() / finite_endpoint_n) if finite_endpoint_n else np.nan
        ),
        "primary_zero_to_zero_transition_n": int(zero_to_zero.sum()),
        "primary_positive_to_zero_transition_n": int(positive_to_zero.sum()),
        "primary_zero_to_positive_transition_n": int(zero_to_positive.sum()),
    }


def analyze_basin(task: tuple) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    gage, source, cfg = task
    frame = pd.read_csv(
        source,
        usecols=[
            "date",
            "q_cfs",
            "qualifier",
            "q_norm",
            "dq_norm_1d",
            "select_p_screened_dry_state_1d",
        ],
        parse_dates=["date"],
        low_memory=False,
    ).sort_values("date")
    context = zero_flow_context(frame)
    context["GAGE_ID"] = gage
    failures: list[dict] = []

    frame["qualifier_next_1d"] = frame["qualifier"].shift(-1)
    frame["q_cfs_next_1d"] = pd.to_numeric(frame["q_cfs"], errors="coerce").shift(-1)
    flag = frame["select_p_screened_dry_state_1d"].fillna(False).astype(bool)
    selected = frame.loc[flag, [
        "date",
        "q_norm",
        "dq_norm_1d",
        "q_cfs",
        "q_cfs_next_1d",
        "qualifier",
        "qualifier_next_1d",
    ]].replace([np.inf, -np.inf], np.nan).dropna(subset=["q_norm", "dq_norm_1d"])
    selected = selected[selected["q_norm"].gt(0)]
    minimum = int(cfg["distribution_test"]["minimum_transitions"])
    if len(selected) < minimum:
        failures.append({
            "GAGE_ID": gage,
            "screen_variant": "primary",
            "status": "excluded_insufficient",
            "reason": f"Only {len(selected)} primary dry-state transitions; need {minimum}",
        })
        return pd.DataFrame(), pd.DataFrame(), context, failures

    rows: list[dict] = []
    profiles: list[pd.DataFrame] = []
    try:
        primary_fit, primary_retained = standardize_like_stage17(selected, cfg)
    except ValueError as error:
        failures.append({
            "GAGE_ID": gage,
            "screen_variant": "primary",
            "status": "excluded_unfittable",
            "reason": str(error),
        })
        return pd.DataFrame(), pd.DataFrame(), context, failures

    primary_row, primary_profile = directional_metrics(
        primary_retained, primary_fit, cfg, "primary", len(selected)
    )
    primary_row["GAGE_ID"] = gage
    primary_profile.insert(0, "GAGE_ID", gage)
    rows.append(primary_row)
    profiles.append(primary_profile)

    origin_estimated = primary_retained["qualifier"].fillna("").astype(str).str.contains(
        ":e", case=False, regex=False
    )
    destination_estimated = primary_retained["qualifier_next_1d"].fillna("").astype(str).str.contains(
        ":e", case=False, regex=False
    )
    nonestimated = primary_retained.loc[
        ~(origin_estimated | destination_estimated)
    ].drop(columns="standardized_innovation")
    sensitivity_minimum = int(
        cfg["distribution_test"]["tail_sensitivity_minimum_transitions"]
    )
    if len(nonestimated) >= sensitivity_minimum:
        try:
            nonestimated_fit, nonestimated_retained = standardize_like_stage17(nonestimated, cfg)
            nonestimated_row, nonestimated_profile = directional_metrics(
                nonestimated_retained,
                nonestimated_fit,
                cfg,
                "nonestimated_endpoints",
                len(nonestimated),
            )
            nonestimated_row["GAGE_ID"] = gage
            nonestimated_profile.insert(0, "GAGE_ID", gage)
            rows.append(nonestimated_row)
            profiles.append(nonestimated_profile)
            nonestimated = nonestimated_retained.drop(columns="standardized_innovation")
        except ValueError as error:
            failures.append({
                "GAGE_ID": gage,
                "screen_variant": "nonestimated_endpoints",
                "status": "sensitivity_unavailable_unfittable",
                "reason": str(error),
            })
    else:
        failures.append({
            "GAGE_ID": gage,
            "screen_variant": "nonestimated_endpoints",
            "status": "sensitivity_unavailable_insufficient",
            "reason": f"Only {len(nonestimated)} transitions; need {sensitivity_minimum}",
        })

    endpoints = pd.concat([
        pd.to_numeric(nonestimated.get("q_cfs"), errors="coerce"),
        pd.to_numeric(nonestimated.get("q_cfs_next_1d"), errors="coerce"),
    ], ignore_index=True)
    positive_endpoints = endpoints[endpoints.gt(0) & np.isfinite(endpoints)]
    cutoff = (
        float(positive_endpoints.quantile(
            float(cfg["distribution_test"]["tail_sensitivity_low_flow_quantile"])
        ))
        if len(positive_endpoints)
        else np.nan
    )
    conservative = nonestimated.copy()
    if np.isfinite(cutoff):
        conservative = conservative[
            pd.to_numeric(conservative["q_cfs"], errors="coerce").gt(cutoff)
            & pd.to_numeric(conservative["q_cfs_next_1d"], errors="coerce").gt(cutoff)
        ]
    if bool(cfg["distribution_test"]["tail_sensitivity_exclude_zero_increments"]):
        conservative = conservative[
            ~np.isclose(
                conservative["dq_norm_1d"].to_numpy(float),
                0.0,
                rtol=0.0,
                atol=0.0,
            )
        ]
    if len(conservative) >= sensitivity_minimum:
        try:
            conservative_fit, conservative_retained = standardize_like_stage17(
                conservative, cfg
            )
            conservative_row, conservative_profile = directional_metrics(
                conservative_retained,
                conservative_fit,
                cfg,
                "conservative",
                len(conservative),
                cutoff,
            )
            conservative_row["GAGE_ID"] = gage
            conservative_profile.insert(0, "GAGE_ID", gage)
            rows.append(conservative_row)
            profiles.append(conservative_profile)
        except ValueError as error:
            failures.append({
                "GAGE_ID": gage,
                "screen_variant": "conservative",
                "status": "sensitivity_unavailable_unfittable",
                "reason": str(error),
            })
    else:
        failures.append({
            "GAGE_ID": gage,
            "screen_variant": "conservative",
            "status": "sensitivity_unavailable_insufficient",
            "reason": f"Only {len(conservative)} transitions; need {sensitivity_minimum}",
        })

    return (
        pd.DataFrame(rows),
        pd.concat(profiles, ignore_index=True),
        context,
        failures,
    )


def summary_row(group: pd.DataFrame, labels: dict) -> dict:
    adequate = group[~group["clustering_low_support"].fillna(True)]
    total_n = int(group["n_standardized"].sum())
    positive_count = int(group["positive_tail_count"].sum())
    negative_count = int(group["negative_tail_count"].sum())

    row = {
        **labels,
        "basin_n": len(group),
        "standardized_transition_n": total_n,
        "pooled_positive_tail_fraction": positive_count / total_n,
        "pooled_negative_tail_fraction": negative_count / total_n,
        "pooled_two_sided_tail_fraction": (positive_count + negative_count) / total_n,
        "median_positive_tail_fraction": float(group["positive_tail_fraction"].median()),
        "q25_positive_tail_fraction": float(group["positive_tail_fraction"].quantile(0.25)),
        "q75_positive_tail_fraction": float(group["positive_tail_fraction"].quantile(0.75)),
        "median_negative_tail_fraction": float(group["negative_tail_fraction"].median()),
        "q25_negative_tail_fraction": float(group["negative_tail_fraction"].quantile(0.25)),
        "q75_negative_tail_fraction": float(group["negative_tail_fraction"].quantile(0.75)),
        "median_two_sided_tail_fraction": float(group["two_sided_tail_fraction"].median()),
        "median_positive_tail_multiple_of_normal": float(
            group["positive_tail_multiple_of_normal"].median()
        ),
        "median_negative_tail_multiple_of_normal": float(
            group["negative_tail_multiple_of_normal"].median()
        ),
        "median_jeffreys_tail_log_count_ratio": float(
            group["jeffreys_tail_log_count_ratio"].median()
        ),
        "basin_fraction_positive_tail_dominant": float(
            (group["positive_tail_count"] > group["negative_tail_count"]).mean()
        ),
        "basin_fraction_negative_tail_dominant": float(
            (group["negative_tail_count"] > group["positive_tail_count"]).mean()
        ),
        "adequate_clustering_support_n": len(adequate),
        "adequate_clustering_support_fraction": len(adequate) / len(group),
        "median_abs_z_lag1_pearson_adequate": float(
            adequate["abs_z_lag1_pearson"].median()
        ),
        "q25_abs_z_lag1_pearson_adequate": float(
            adequate["abs_z_lag1_pearson"].quantile(0.25)
        ),
        "q75_abs_z_lag1_pearson_adequate": float(
            adequate["abs_z_lag1_pearson"].quantile(0.75)
        ),
        "median_abs_z_lag1_spearman_adequate": float(
            adequate["abs_z_lag1_spearman"].median()
        ),
        "median_z2_lag1_pearson_adequate": float(
            adequate["z2_lag1_pearson"].median()
        ),
        "median_z2_lag1_spearman_adequate": float(
            adequate["z2_lag1_spearman"].median()
        ),
        "fraction_abs_z_lag1_pearson_positive_adequate": float(
            adequate["abs_z_lag1_pearson"].gt(0).mean()
        ) if len(adequate) else np.nan,
        "fraction_z2_lag1_pearson_positive_adequate": float(
            adequate["z2_lag1_pearson"].gt(0).mean()
        ) if len(adequate) else np.nan,
    }
    return row


def grouped_summaries(basin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    continental_rows = []
    ecoregion_rows = []
    for variant, group in basin.groupby("screen_variant", sort=False):
        continental_rows.append(summary_row(group, {"screen_variant": variant}))
        for ecoregion, regional in group.groupby("AGGECOREGION", dropna=False):
            ecoregion_rows.append(summary_row(
                regional,
                {
                    "screen_variant": variant,
                    "AGGECOREGION": ecoregion,
                },
            ))
    return pd.DataFrame(continental_rows), pd.DataFrame(ecoregion_rows)


def profile_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["screen_variant", "threshold_abs_z"]
    for key, group in profile.groupby(keys, sort=False):
        variant, threshold = key
        total = int(group["n_standardized"].sum())
        pos = int(group["positive_tail_count"].sum())
        neg = int(group["negative_tail_count"].sum())
        rows.append({
            "screen_variant": variant,
            "threshold_abs_z": threshold,
            "basin_n": len(group),
            "standardized_transition_n": total,
            "pooled_positive_tail_fraction": pos / total,
            "pooled_negative_tail_fraction": neg / total,
            "median_basin_positive_tail_fraction": float(
                group["positive_tail_fraction"].median()
            ),
            "q25_basin_positive_tail_fraction": float(
                group["positive_tail_fraction"].quantile(0.25)
            ),
            "q75_basin_positive_tail_fraction": float(
                group["positive_tail_fraction"].quantile(0.75)
            ),
            "median_basin_negative_tail_fraction": float(
                group["negative_tail_fraction"].median()
            ),
            "q25_basin_negative_tail_fraction": float(
                group["negative_tail_fraction"].quantile(0.25)
            ),
            "q75_basin_negative_tail_fraction": float(
                group["negative_tail_fraction"].quantile(0.75)
            ),
            "normal_one_sided_tail_reference": float(
                group["normal_one_sided_tail_reference"].iloc[0]
            ),
        })
    return pd.DataFrame(rows)


def validate_against_stage17(
    basin: pd.DataFrame, stage17: pd.DataFrame, require_complete: bool
) -> tuple[pd.DataFrame, dict]:
    ours = basin.copy()
    ours_validation = ours[[
        "GAGE_ID",
        "screen_variant",
        "n_standardized",
        "variance_exponent",
        "variance_coefficient",
        "variance_r2",
        "two_sided_tail_fraction",
    ]]
    frozen_parts = []
    variant_columns = {
        "primary": {
            "n_standardized": "n_standardized",
            "variance_exponent": "variance_exponent",
            "variance_coefficient": "variance_coefficient",
            "variance_r2": "variance_r2",
            "standardized_tail_fraction": "standardized_tail_fraction",
        },
        "nonestimated_endpoints": {
            "n_standardized": "nonestimated_n",
            "variance_exponent": "nonestimated_variance_exponent",
            "variance_r2": "nonestimated_variance_r2",
            "standardized_tail_fraction": "nonestimated_standardized_tail_fraction",
        },
        "conservative": {
            "n_standardized": "conservative_n",
            "variance_exponent": "conservative_variance_exponent",
            "variance_r2": "conservative_variance_r2",
            "standardized_tail_fraction": "conservative_standardized_tail_fraction",
        },
    }
    for variant, mapping in variant_columns.items():
        columns = ["GAGE_ID", *mapping.values()]
        frozen_variant = stage17[columns].rename(
            columns={source: target for target, source in mapping.items()}
        )
        if variant != "primary":
            frozen_variant = frozen_variant.dropna(subset=["variance_exponent"])
            frozen_variant["variance_coefficient"] = np.nan
        frozen_variant["screen_variant"] = variant
        frozen_parts.append(frozen_variant)
    frozen_validation = pd.concat(frozen_parts, ignore_index=True)

    compare = ours_validation.merge(
        frozen_validation,
        on=["GAGE_ID", "screen_variant"],
        how="outer" if require_complete else "left",
        suffixes=("_new", "_stage17"),
        indicator=True,
    )
    compare["n_standardized_match"] = (
        compare["n_standardized_new"] == compare["n_standardized_stage17"]
    )
    pairs = {
        "variance_exponent": ("variance_exponent_new", "variance_exponent_stage17"),
        "variance_coefficient": ("variance_coefficient_new", "variance_coefficient_stage17"),
        "variance_r2": ("variance_r2_new", "variance_r2_stage17"),
        "tail_fraction": ("two_sided_tail_fraction", "standardized_tail_fraction"),
    }
    for label, (new, old) in pairs.items():
        compare[f"{label}_abs_difference"] = (compare[new] - compare[old]).abs()
        compare[f"{label}_match"] = np.isclose(
            compare[new], compare[old], rtol=FLOAT_RTOL, atol=FLOAT_ATOL, equal_nan=True
        )
    compare["variance_coefficient_match"] |= compare["screen_variant"].ne("primary")
    match_columns = ["n_standardized_match"] + [f"{key}_match" for key in pairs]
    compare["all_metrics_match"] = compare[match_columns].all(axis=1) & compare["_merge"].eq("both")
    primary_new = ours[ours["screen_variant"].eq("primary")]
    per_variant = {}
    for variant, group in compare.groupby("screen_variant", sort=False):
        per_variant[str(variant)] = {
            "new_rows": int(group["_merge"].isin(["both", "left_only"]).sum()),
            "frozen_rows": int(group["_merge"].isin(["both", "right_only"]).sum()),
            "matched_rows": int(group["_merge"].eq("both").sum()),
            "all_metrics_match": bool(group["all_metrics_match"].all()),
        }
    summary = {
        "new_primary_basin_n": len(primary_new),
        "frozen_stage17_basin_n": len(stage17),
        "new_variant_row_n": len(ours_validation),
        "frozen_variant_row_n": len(frozen_validation),
        "matched_variant_row_n": int(compare["_merge"].eq("both").sum()),
        "new_only_variant_row_n": int(compare["_merge"].eq("left_only").sum()),
        "stage17_only_variant_row_n": int(compare["_merge"].eq("right_only").sum()),
        "all_metrics_match_n": int(compare["all_metrics_match"].sum()),
        "all_metrics_match": bool(compare["all_metrics_match"].all()),
        "per_variant": per_variant,
        "rtol": FLOAT_RTOL,
        "atol": FLOAT_ATOL,
    }
    for key in pairs:
        values = compare[f"{key}_abs_difference"]
        summary[f"maximum_{key}_abs_difference"] = (
            float(values.max()) if values.notna().any() else 0.0
        )
    return compare, summary


def methods_text() -> str:
    return """# Signed-tail and within-spell clustering sensitivity

This isolated sensitivity analysis reads the frozen Stage-10 daily transition
files. It does not alter Stages 01--20.

For each basin and screen variant, the conditional variance law and innovation
standardization reproduce Stage 17 exactly. Conditional mean increments are
interpolated over log flow from the Stage-17 state bins. With fitted
`D2(q) = a q^m`, the preliminary residual is

`r_t = [Delta q_t - E(Delta q | q_t)] / sqrt(2 a q_t^m)`.

The final innovation is `z_t = (r_t - mean(r)) / sd(r)`, where the center and
scale are estimated separately inside each screen variant.

At the prespecified threshold `c = 3`, directional frequencies are
`p_plus = N(z > c)/N` and `p_minus = N(z < -c)/N`. Raw differences are
accompanied by the stable Jeffreys count ratio
`log[(N_plus + 1/2)/(N_minus + 1/2)]`; the half-count avoids infinities when
one directional tail has no observations. Jeffreys 95% intervals are stored
for each directional proportion.

Lag-one clustering never bridges gaps between screened observations. A pair
is used only when two selected one-day transition origins occur on consecutive
calendar dates, which keeps both transitions in the same screened dry spell.
Pearson and Spearman correlations are reported for `|z_t|` versus `|z_(t+1)|`
and `z_t^2` versus `z_(t+1)^2`. Basins with fewer than 100 such pairs are
explicitly marked as low support; correlations with fewer than three pairs or
constant inputs are undefined. Because adjacent daily increments share their
middle discharge endpoint, these are descriptive persistence diagnostics and
are not interpreted as an independently identified GARCH process.

Variants are: (1) the primary precipitation- and proxy-snowmelt-screened dry
state sample; (2) removal of transitions with an estimated-value qualifier at
either endpoint; and (3) that endpoint screen plus removal of the lowest 1% of
positive endpoint discharges and exact-zero increments. Variant-specific laws
and standardizations are refitted, as in Stage 17.

The zero-flow context file covers every accepted Stage-10 basin, including
basins that cannot support the Stage-17 estimator. Exact zeros are summarized
descriptively; this analysis does not fit a hurdle model.
"""


def main() -> None:
    args = parse_arguments()
    cfg = load_config(args.config)
    out = (
        Path(args.output).expanduser().resolve()
        if args.output
        else output_root(cfg) / "sensitivity_analyses" / "tail_direction"
    )
    receipt_path = out / "_SUCCESS.json"
    if receipt_path.exists() and not args.force:
        raise RuntimeError(f"Outputs already complete at {out}; use --force to replace them")
    out.mkdir(parents=True, exist_ok=True)
    if args.force:
        receipt_path.unlink(missing_ok=True)

    stage10_receipt = stage_dir(cfg, 10, create=False) / "_SUCCESS.json"
    stage15_receipt = stage_dir(cfg, 15, create=False) / "_SUCCESS.json"
    stage17_receipt = stage_dir(cfg, 17, create=False) / "_SUCCESS.json"
    required_receipts = (stage10_receipt, stage15_receipt, stage17_receipt)
    missing_receipts = [str(path) for path in required_receipts if not path.exists()]
    if missing_receipts:
        raise RuntimeError(
            "Frozen Stage-10, Stage-15, and Stage-17 receipts are required: "
            + ", ".join(missing_receipts)
        )

    inventory = pd.read_csv(
        stage_dir(cfg, 10, create=False) / "transition_inventory.csv",
        dtype={"GAGE_ID": str},
    )
    inventory["GAGE_ID"] = inventory["GAGE_ID"].astype(str).str.zfill(8)
    inventory = inventory.sort_values("GAGE_ID")
    if args.limit:
        inventory = inventory.head(args.limit)
    daily_dir = stage_dir(cfg, 10, create=False) / "daily"
    tasks = [
        (row.GAGE_ID, str(daily_dir / f"{row.GAGE_ID}.csv.gz"), cfg)
        for row in inventory.itertuples(index=False)
    ]

    basin_parts: list[pd.DataFrame] = []
    profile_parts: list[pd.DataFrame] = []
    contexts: list[dict] = []
    failures: list[dict] = []
    unexpected = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(analyze_basin, task): task[0] for task in tasks}
        for number, future in enumerate(as_completed(futures), start=1):
            gage = futures[future]
            try:
                basin_part, profile_part, context, basin_failures = future.result()
                if not basin_part.empty:
                    basin_parts.append(basin_part)
                    profile_parts.append(profile_part)
                contexts.append(context)
                failures.extend(basin_failures)
            except Exception as error:  # preserve a complete, auditable failure inventory
                unexpected += 1
                failures.append({
                    "GAGE_ID": gage,
                    "screen_variant": "all",
                    "status": "failed_unexpected",
                    "reason": repr(error),
                })
            if number % 250 == 0 or number == len(tasks):
                print(f"Completed {number}/{len(tasks)} basins", flush=True)

    if not basin_parts:
        raise RuntimeError("No basin supported directional-tail diagnostics")
    basin = pd.concat(basin_parts, ignore_index=True)
    profile = pd.concat(profile_parts, ignore_index=True)
    context = pd.DataFrame(contexts)
    failures_frame = pd.DataFrame(
        failures, columns=["GAGE_ID", "screen_variant", "status", "reason"]
    )

    attributes = pd.read_csv(
        stage_dir(cfg, 15, create=False) / "continental_basin_results.csv",
        dtype={"GAGE_ID": str},
        low_memory=False,
    )
    attributes["GAGE_ID"] = attributes["GAGE_ID"].astype(str).str.zfill(8)
    attribute_columns = [
        column for column in [
            "GAGE_ID",
            "quality_tier",
            "is_reference",
            "AGGECOREGION",
            "ecoregion",
            "LAT_GAGE",
            "LNG_GAGE",
            "LAT_CENT",
            "LONG_CENT",
            "aridity",
            "BFI_AVE",
            "log_area",
            "FLOW_PCT_EST_VALUES",
            "analysis_population",
        ] if column in attributes.columns
    ]
    attribute_subset = attributes[attribute_columns].drop_duplicates("GAGE_ID")
    basin = basin.merge(attribute_subset, on="GAGE_ID", how="left", validate="many_to_one")
    profile = profile.merge(
        attribute_subset[[
            column for column in ["GAGE_ID", "AGGECOREGION", "is_reference"]
            if column in attribute_subset
        ]],
        on="GAGE_ID",
        how="left",
        validate="many_to_one",
    )
    basin = basin.sort_values(["GAGE_ID", "screen_variant"]).reset_index(drop=True)
    profile_sort = [
        column for column in ("GAGE_ID", "screen_variant", "threshold")
        if column in profile.columns
    ]
    profile = profile.sort_values(profile_sort).reset_index(drop=True)
    context = context.sort_values("GAGE_ID").reset_index(drop=True)
    failures_frame = failures_frame.sort_values(
        ["GAGE_ID", "screen_variant", "status"]
    ).reset_index(drop=True)

    frozen = pd.read_csv(
        stage_dir(cfg, 17, create=False) / "distribution_basin_diagnostics.csv",
        dtype={"GAGE_ID": str},
    )
    frozen["GAGE_ID"] = frozen["GAGE_ID"].astype(str).str.zfill(8)
    estimable = set(frozen["GAGE_ID"])
    context["stage17_primary_estimable"] = context["GAGE_ID"].isin(estimable)
    context = context.merge(attribute_subset, on="GAGE_ID", how="left", validate="one_to_one")

    require_complete = args.limit is None
    validation, validation_summary = validate_against_stage17(
        basin, frozen, require_complete=require_complete
    )
    continental, ecoregion = grouped_summaries(basin)
    profile_continental = profile_summary(profile)

    paths = {
        "basin": out / "basin_tail_direction_and_clustering.csv",
        "profile": out / "basin_tail_exceedance_profile.csv.gz",
        "profile_continental": out / "continental_tail_exceedance_profile.csv",
        "continental": out / "continental_summary.csv",
        "ecoregion": out / "ecoregion_summary.csv",
        "zero_flow": out / "basin_zero_flow_context.csv",
        "failures": out / "analysis_exclusions_and_failures.csv",
        "validation": out / "validation_against_stage17.csv",
        "validation_summary": out / "VALIDATION_SUMMARY.json",
        "methods": out / "METHODS.md",
    }
    write_csv_atomic(basin.sort_values(["screen_variant", "GAGE_ID"]), paths["basin"])
    write_csv_atomic(
        profile.sort_values(["screen_variant", "GAGE_ID", "threshold_abs_z"]),
        paths["profile"],
        compression="gzip",
    )
    write_csv_atomic(profile_continental, paths["profile_continental"])
    write_csv_atomic(continental, paths["continental"])
    write_csv_atomic(ecoregion, paths["ecoregion"])
    write_csv_atomic(context.sort_values("GAGE_ID"), paths["zero_flow"])
    write_csv_atomic(failures_frame.sort_values(["GAGE_ID", "screen_variant"]), paths["failures"])
    write_csv_atomic(validation.sort_values("GAGE_ID"), paths["validation"])
    atomic_json(paths["validation_summary"], validation_summary)
    write_text_atomic(methods_text(), paths["methods"])

    if unexpected:
        raise RuntimeError(f"{unexpected} unexpected basin failures; inspect {paths['failures']}")
    if not validation_summary["all_metrics_match"]:
        raise RuntimeError(
            "New primary standardization does not reproduce frozen Stage 17; "
            f"inspect {paths['validation']}"
        )

    primary = basin[basin["screen_variant"].eq("primary")]
    metrics = {
        "basins_requested": len(tasks),
        "primary_basins_analyzed": len(primary),
        "all_variant_rows": len(basin),
        "zero_flow_context_basins": len(context),
        "unexpected_failures": unexpected,
        "stage17_validation_passed": validation_summary["all_metrics_match"],
        "primary_median_positive_tail_fraction": float(
            primary["positive_tail_fraction"].median()
        ),
        "primary_median_negative_tail_fraction": float(
            primary["negative_tail_fraction"].median()
        ),
        "primary_fraction_positive_tail_dominant": float(
            (primary["positive_tail_count"] > primary["negative_tail_count"]).mean()
        ),
        "primary_adequate_clustering_support_fraction": float(
            (~primary["clustering_low_support"]).mean()
        ),
        "primary_median_abs_z_lag1_pearson_adequate": float(
            primary.loc[
                ~primary["clustering_low_support"], "abs_z_lag1_pearson"
            ].median()
        ),
        "primary_median_z2_lag1_pearson_adequate": float(
            primary.loc[
                ~primary["clustering_low_support"], "z2_lag1_pearson"
            ].median()
        ),
    }
    script_path = Path(__file__).resolve()
    receipt = {
        "analysis": "tail_direction_and_clustering_sensitivity",
        "analysis_version": ANALYSIS_VERSION,
        "completed_utc": utc_now(),
        "config": str(Path(cfg["_config_path"]).resolve()),
        "config_sha256": sha256_file(Path(cfg["_config_path"])),
        "code_sha256": {str(script_path.relative_to(Path(cfg["_project_root"]))): sha256_file(script_path)},
        "frozen_input_receipts_sha256": {
            str(stage10_receipt.relative_to(Path(cfg["_project_root"]))): sha256_file(stage10_receipt),
            str(stage15_receipt.relative_to(Path(cfg["_project_root"]))): sha256_file(stage15_receipt),
            str(stage17_receipt.relative_to(Path(cfg["_project_root"]))): sha256_file(stage17_receipt),
        },
        "frozen_stage15_attributes_sha256": sha256_file(
            stage_dir(cfg, 15, create=False) / "continental_basin_results.csv"
        ),
        "frozen_stage17_diagnostics_sha256": sha256_file(
            stage_dir(cfg, 17, create=False) / "distribution_basin_diagnostics.csv"
        ),
        "outputs": [str(path.resolve()) for path in paths.values()],
        "output_sha256": {
            str(path.name): sha256_file(path) for path in paths.values()
        },
        "metrics": metrics,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
