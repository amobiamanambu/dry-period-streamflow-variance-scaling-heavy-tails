#!/usr/bin/env python3
"""Sensitivity analysis for robust dry-period scale exponents.

This isolated, versioned analysis reads the frozen Stage-10 transition files
for the Stage-17 estimable population. It reproduces the primary binned
second-moment exponent before fitting variance-equivalent exponents from the
squared, Gaussian-consistent median absolute deviation (MAD) and interquartile
range (IQR). It never writes to the numbered workflow stages.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from lib.common import (
    DEFAULT_CONFIG,
    load_config,
    output_root,
    sha256_file,
    stage_dir,
    utc_now,
)
from lib.statistics import benjamini_hochberg
from lib.stochastic import loglog_power_fit


ANALYSIS_VERSION = 1
PRIMARY_SCREEN_COLUMN = "select_p_screened_dry_state_1d"
PRIMARY_LAG_DAYS = 1
MAD_NORMALIZER = float(stats.norm.ppf(0.75))
IQR_NORMALIZER = 2.0 * MAD_NORMALIZER
VALIDATION_RTOL = 1e-10
VALIDATION_ATOL = 1e-12
DEFAULT_BOOTSTRAP_RESAMPLES = 20_000
DEFAULT_BOOTSTRAP_SEED = 20260909

ESTIMATORS = (
    (
        "ols",
        "Primary centered second moment",
        "conditional_variance_rate",
        "ols_variance_exponent",
    ),
    (
        "mad2",
        "Squared Gaussian-consistent MAD",
        "mad2_variance_equivalent_rate",
        "mad2_variance_equivalent_exponent",
    ),
    (
        "iqr2",
        "Squared Gaussian-consistent IQR",
        "iqr2_variance_equivalent_rate",
        "iqr2_variance_equivalent_exponent",
    ),
)

TAIL_DESCRIPTORS = (
    ("two_sided_tail_multiple_of_normal", "Two-sided |z| > 3 multiple"),
    ("positive_tail_multiple_of_normal", "Positive z > 3 multiple"),
    ("negative_tail_multiple_of_normal", "Negative z < -3 multiple"),
    ("jeffreys_tail_log_count_ratio", "Signed positive/negative tail log ratio"),
    ("absolute_tail_log_count_ratio", "Absolute directional tail log ratio"),
    ("standardized_excess_kurtosis", "Standardized excess kurtosis"),
)

RELATIONSHIP_OUTCOMES = (
    ("ols_variance_exponent", "Primary variance exponent"),
    ("mad2_variance_equivalent_exponent", "Squared-MAD exponent"),
    ("iqr2_variance_equivalent_exponent", "Squared-IQR exponent"),
    ("mad2_minus_ols", "Squared-MAD minus primary exponent"),
    ("iqr2_minus_ols", "Squared-IQR minus primary exponent"),
    ("absolute_mad2_minus_ols", "Absolute squared-MAD difference"),
    ("absolute_iqr2_minus_ols", "Absolute squared-IQR difference"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_csv_atomic(
    frame: pd.DataFrame,
    path: Path,
    *,
    compression: str | None = None,
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.15g",
        compression=compression,
    )
    temporary.replace(path)


def write_json_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_text_atomic(value: str, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def truthy(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def build_bin_statistics(
    selected: pd.DataFrame,
    state_bins: int,
    minimum_bin_count: int,
) -> pd.DataFrame:
    q = pd.to_numeric(selected["q_norm"], errors="coerce").to_numpy(float)
    dq = pd.to_numeric(selected["dq_norm_1d"], errors="coerce").to_numpy(float)
    valid = np.isfinite(q) & np.isfinite(dq) & (q > 0)
    q, dq = q[valid], dq[valid]
    if len(q) < state_bins * minimum_bin_count:
        return pd.DataFrame()

    edges = np.unique(np.quantile(q, np.linspace(0.0, 1.0, state_bins + 1)))
    if len(edges) < 4:
        return pd.DataFrame()
    assignments = np.searchsorted(edges[1:-1], q, side="right")
    rows: list[dict] = []
    for state in range(len(edges) - 1):
        in_bin = assignments == state
        n = int(in_bin.sum())
        if n < minimum_bin_count:
            continue
        q_bin = q[in_bin]
        increments = dq[in_bin]
        mean_increment = float(np.mean(increments))
        centered = increments - mean_increment
        variance_rate = float(np.mean(centered**2) / (2.0 * PRIMARY_LAG_DAYS))
        median_increment = float(np.median(increments))
        raw_mad = float(np.median(np.abs(increments - median_increment)))
        q25, q75 = np.quantile(increments, (0.25, 0.75))
        raw_iqr = float(q75 - q25)
        mad_sigma = raw_mad / MAD_NORMALIZER
        iqr_sigma = raw_iqr / IQR_NORMALIZER
        rows.append({
            "state_bin": int(state),
            "q_center": float(np.exp(np.mean(np.log(q_bin)))),
            "n": n,
            "conditional_mean_increment": mean_increment,
            "conditional_median_increment": median_increment,
            "conditional_variance_rate": variance_rate,
            "raw_mad": raw_mad,
            "raw_iqr": raw_iqr,
            "gaussian_consistent_mad_scale": mad_sigma,
            "gaussian_consistent_iqr_scale": iqr_sigma,
            "mad2_variance_equivalent_rate": float(
                mad_sigma**2 / (2.0 * PRIMARY_LAG_DAYS)
            ),
            "iqr2_variance_equivalent_rate": float(
                iqr_sigma**2 / (2.0 * PRIMARY_LAG_DAYS)
            ),
        })
    return pd.DataFrame(rows)


def fit_scale_law(
    bins: pd.DataFrame,
    response: str,
    minimum_valid_bins: int,
) -> dict:
    if bins.empty:
        return loglog_power_fit(np.array([]), np.array([]), minimum_valid_bins)
    return loglog_power_fit(
        bins["q_center"].to_numpy(float),
        bins[response].to_numpy(float),
        minimum_valid_bins,
    )


def analyze_basin(task: tuple) -> tuple[dict, pd.DataFrame, list[dict]]:
    (
        gage,
        source,
        expected_n,
        state_bins,
        minimum_bin_count,
        minimum_valid_bins,
    ) = task
    row: dict = {
        "analysis_version": ANALYSIS_VERSION,
        "GAGE_ID": gage,
        "expected_stage17_transitions": int(expected_n),
        "n_selected": 0,
        "n_bins_total": 0,
        "robust_scale_status": "unavailable",
    }
    failures: list[dict] = []
    try:
        frame = pd.read_csv(
            source,
            usecols=["q_norm", "dq_norm_1d", PRIMARY_SCREEN_COLUMN],
            low_memory=False,
        )
        flag = truthy(frame[PRIMARY_SCREEN_COLUMN])
        selected = frame.loc[flag, ["q_norm", "dq_norm_1d"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        selected = selected[pd.to_numeric(selected["q_norm"], errors="coerce").gt(0)]
        row["n_selected"] = int(len(selected))
        bins = build_bin_statistics(selected, state_bins, minimum_bin_count)
        row["n_bins_total"] = int(len(bins))
        if bins.empty:
            raise ValueError("No valid equal-frequency state bins")

        for key, _, response, _ in ESTIMATORS:
            fit = fit_scale_law(bins, response, minimum_valid_bins)
            row[f"{key}_variance_exponent"] = fit["exponent"]
            row[f"{key}_variance_coefficient"] = fit["coefficient"]
            row[f"{key}_variance_r2"] = fit["r2"]
            row[f"{key}_valid_bins"] = fit["n_bins_fit"]
            row[f"{key}_slope_se"] = fit.get("slope_se", np.nan)

        row["zero_mad_scale_bins"] = int(
            bins["mad2_variance_equivalent_rate"].le(0).sum()
        )
        row["zero_iqr_scale_bins"] = int(
            bins["iqr2_variance_equivalent_rate"].le(0).sum()
        )
        mad_available = np.isfinite(row["mad2_variance_exponent"])
        iqr_available = np.isfinite(row["iqr2_variance_exponent"])
        row["robust_scale_status"] = (
            "both_available"
            if mad_available and iqr_available
            else "mad_only"
            if mad_available
            else "iqr_only"
            if iqr_available
            else "both_unavailable"
        )
        if not mad_available:
            failures.append({
                "GAGE_ID": gage,
                "component": "mad2",
                "status": "robust_estimator_unavailable",
                "reason": (
                    f"Only {row['mad2_valid_bins']} positive squared-MAD bins; "
                    f"need {minimum_valid_bins}"
                ),
            })
        if not iqr_available:
            failures.append({
                "GAGE_ID": gage,
                "component": "iqr2",
                "status": "robust_estimator_unavailable",
                "reason": (
                    f"Only {row['iqr2_valid_bins']} positive squared-IQR bins; "
                    f"need {minimum_valid_bins}"
                ),
            })
        bins.insert(0, "GAGE_ID", gage)
        bins.insert(0, "analysis_version", ANALYSIS_VERSION)
        return row, bins, failures
    except Exception as error:
        failures.append({
            "GAGE_ID": gage,
            "component": "basin",
            "status": "unexpected_failure",
            "reason": repr(error),
        })
        for key, _, _, _ in ESTIMATORS:
            row.setdefault(f"{key}_variance_exponent", np.nan)
            row.setdefault(f"{key}_variance_coefficient", np.nan)
            row.setdefault(f"{key}_variance_r2", np.nan)
            row.setdefault(f"{key}_valid_bins", 0)
            row.setdefault(f"{key}_slope_se", np.nan)
        row.setdefault("zero_mad_scale_bins", 0)
        row.setdefault("zero_iqr_scale_bins", 0)
        return row, pd.DataFrame(), failures


def bootstrap_median_interval(
    values: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or resamples <= 0:
        return np.nan, np.nan
    medians: list[np.ndarray] = []
    for start in range(0, resamples, 256):
        size = min(256, resamples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        medians.append(np.median(values[indices], axis=1))
    return tuple(
        float(value) for value in np.quantile(np.concatenate(medians), (0.025, 0.975))
    )


def distribution_fields(values: pd.Series, prefix: str = "") -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    key = f"{prefix}_" if prefix else ""
    if not len(x):
        return {
            f"{key}n": 0,
            f"{key}median": np.nan,
            f"{key}q25": np.nan,
            f"{key}q75": np.nan,
        }
    q25, median, q75 = np.quantile(x, (0.25, 0.5, 0.75))
    return {
        f"{key}n": int(len(x)),
        f"{key}median": float(median),
        f"{key}q25": float(q25),
        f"{key}q75": float(q75),
    }


def population_summaries(
    basin: pd.DataFrame,
    resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = np.logical_and.reduce([
        np.isfinite(pd.to_numeric(basin[column], errors="coerce"))
        for _, _, _, column in ESTIMATORS
    ])
    rng = np.random.default_rng(seed)
    estimator_rows: list[dict] = []
    for key, label, _, column in ESTIMATORS:
        values = pd.to_numeric(basin[column], errors="coerce")
        low, high = bootstrap_median_interval(values.to_numpy(float), resamples, rng)
        common_values = values.loc[common]
        common_low, common_high = bootstrap_median_interval(
            common_values.to_numpy(float), resamples, rng
        )
        estimator_rows.append({
            "estimator_key": key,
            "estimator": label,
            **distribution_fields(values),
            "bootstrap_median_ci_low": low,
            "bootstrap_median_ci_high": high,
            "common_complete_case_n": int(common.sum()),
            **distribution_fields(common_values, "common"),
            "common_bootstrap_median_ci_low": common_low,
            "common_bootstrap_median_ci_high": common_high,
        })

    comparison_rows: list[dict] = []
    primary = pd.to_numeric(basin["ols_variance_exponent"], errors="coerce")
    for key, label, _, column in ESTIMATORS[1:]:
        robust = pd.to_numeric(basin[column], errors="coerce")
        paired = pd.DataFrame({"primary": primary, "robust": robust}).dropna()
        delta = paired["robust"] - paired["primary"]
        low, high = bootstrap_median_interval(delta.to_numpy(float), resamples, rng)
        rho, p_value = stats.spearmanr(paired["primary"], paired["robust"])
        comparison_rows.append({
            "comparison_key": f"{key}_minus_ols",
            "robust_estimator": label,
            "paired_n": int(len(paired)),
            "median_delta_robust_minus_primary": float(delta.median()),
            "q25_delta_robust_minus_primary": float(delta.quantile(0.25)),
            "q75_delta_robust_minus_primary": float(delta.quantile(0.75)),
            "bootstrap_median_delta_ci_low": low,
            "bootstrap_median_delta_ci_high": high,
            "median_absolute_delta": float(delta.abs().median()),
            "p90_absolute_delta": float(delta.abs().quantile(0.90)),
            "fraction_absolute_delta_le_0p25": float(delta.abs().le(0.25).mean()),
            "fraction_robust_above_primary": float(delta.gt(0).mean()),
            "spearman_rho_vs_primary": float(rho),
            "spearman_p_value": float(p_value),
            "classification_relative_to_2_agreement": float(
                (np.sign(paired["robust"] - 2.0) == np.sign(paired["primary"] - 2.0)).mean()
            ),
        })
    return pd.DataFrame(estimator_rows), pd.DataFrame(comparison_rows)


def tail_relationships(basin: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for outcome, outcome_label in RELATIONSHIP_OUTCOMES:
        for descriptor, descriptor_label in TAIL_DESCRIPTORS:
            pair = basin[[outcome, descriptor]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if len(pair) < 20 or pair[outcome].nunique() < 2 or pair[descriptor].nunique() < 2:
                rho, p_value = np.nan, np.nan
            else:
                result = stats.spearmanr(pair[outcome], pair[descriptor])
                rho, p_value = float(result.statistic), float(result.pvalue)
            rows.append({
                "outcome": outcome,
                "outcome_label": outcome_label,
                "tail_descriptor": descriptor,
                "tail_descriptor_label": descriptor_label,
                "paired_n": int(len(pair)),
                "spearman_rho": rho,
                "p_value": p_value,
            })
    summary = pd.DataFrame(rows)
    summary["p_fdr_bh"] = benjamini_hochberg(summary["p_value"])
    return summary


def tail_quintile_summaries(basin: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    outcomes = [
        "ols_variance_exponent",
        "mad2_variance_equivalent_exponent",
        "iqr2_variance_equivalent_exponent",
        "mad2_minus_ols",
        "iqr2_minus_ols",
    ]
    for descriptor, descriptor_label in TAIL_DESCRIPTORS:
        valid = basin[[descriptor, *outcomes]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna(subset=[descriptor]).copy()
        if len(valid) < 20 or valid[descriptor].nunique() < 5:
            continue
        valid["tail_quintile"] = pd.qcut(
            valid[descriptor], 5, labels=False, duplicates="drop"
        ) + 1
        for quintile, group in valid.groupby("tail_quintile", observed=True):
            row = {
                "tail_descriptor": descriptor,
                "tail_descriptor_label": descriptor_label,
                "tail_quintile": int(quintile),
                "basin_n": int(len(group)),
                "descriptor_q25": float(group[descriptor].quantile(0.25)),
                "descriptor_median": float(group[descriptor].median()),
                "descriptor_q75": float(group[descriptor].quantile(0.75)),
            }
            for outcome in outcomes:
                row[f"median_{outcome}"] = float(group[outcome].median())
                row[f"q25_{outcome}"] = float(group[outcome].quantile(0.25))
                row[f"q75_{outcome}"] = float(group[outcome].quantile(0.75))
            rows.append(row)
    return pd.DataFrame(rows)


def ecoregion_summaries(basin: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for region, group in basin.groupby("AGGECOREGION", dropna=False):
        rows.append({
            "AGGECOREGION": str(region),
            "basin_n": int(len(group)),
            **distribution_fields(group["ols_variance_exponent"], "ols"),
            **distribution_fields(group["mad2_variance_equivalent_exponent"], "mad2"),
            **distribution_fields(group["iqr2_variance_equivalent_exponent"], "iqr2"),
            **distribution_fields(group["mad2_minus_ols"], "mad2_minus_ols"),
            **distribution_fields(group["iqr2_minus_ols"], "iqr2_minus_ols"),
        })
    return pd.DataFrame(rows).sort_values("AGGECOREGION").reset_index(drop=True)


def validate_against_stage17(
    basin: pd.DataFrame,
    stage17: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    frozen = stage17[[
        "GAGE_ID",
        "n_transitions",
        "variance_exponent",
        "variance_coefficient",
        "variance_r2",
    ]].rename(columns={
        "n_transitions": "n_selected_stage17",
        "variance_exponent": "ols_variance_exponent_stage17",
        "variance_coefficient": "ols_variance_coefficient_stage17",
        "variance_r2": "ols_variance_r2_stage17",
    })
    ours = basin[[
        "GAGE_ID",
        "n_selected",
        "ols_variance_exponent",
        "ols_variance_coefficient",
        "ols_variance_r2",
    ]]
    compare = ours.merge(
        frozen,
        on="GAGE_ID",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    compare["n_selected_match"] = compare["n_selected"].eq(compare["n_selected_stage17"])
    numeric_fields = (
        "ols_variance_exponent",
        "ols_variance_coefficient",
        "ols_variance_r2",
    )
    for field in numeric_fields:
        new = pd.to_numeric(compare[field], errors="coerce")
        old = pd.to_numeric(compare[f"{field}_stage17"], errors="coerce")
        compare[f"{field}_abs_difference"] = (new - old).abs()
        compare[f"{field}_match"] = np.isclose(
            new,
            old,
            rtol=VALIDATION_RTOL,
            atol=VALIDATION_ATOL,
            equal_nan=True,
        )
    match_fields = ["n_selected_match", *[f"{field}_match" for field in numeric_fields]]
    compare["all_metrics_match"] = (
        compare["_merge"].eq("both") & compare[match_fields].all(axis=1)
    )
    summary = {
        "new_basin_n": int(len(ours)),
        "stage17_basin_n": int(len(frozen)),
        "matched_basin_n": int(compare["_merge"].eq("both").sum()),
        "new_only_basin_n": int(compare["_merge"].eq("left_only").sum()),
        "stage17_only_basin_n": int(compare["_merge"].eq("right_only").sum()),
        "n_selected_match_n": int(compare["n_selected_match"].sum()),
        "all_metrics_match_n": int(compare["all_metrics_match"].sum()),
        "all_metrics_match": bool(compare["all_metrics_match"].all()),
        "rtol": VALIDATION_RTOL,
        "atol": VALIDATION_ATOL,
    }
    for field in numeric_fields:
        differences = compare[f"{field}_abs_difference"]
        summary[f"maximum_{field}_abs_difference"] = (
            float(differences.max()) if differences.notna().any() else 0.0
        )
    return compare, summary


def methods_text(settings: dict, resamples: int, seed: int) -> str:
    return f"""# Robust-scale exponent sensitivity

This isolated sensitivity analysis reads the frozen Stage-10 one-day transition
files for the {PRIMARY_SCREEN_COLUMN} population that was estimable in Stage
17. It does not alter the numbered workflow stages.

The sample, flow normalization, 20 equal-frequency starting-flow bins,
minimum count of {settings['minimum_bin_count']} transitions per bin, and
minimum {settings['minimum_valid_bins_for_fit']} positive bins are identical
to the primary conditional-variance analysis. Before robust results are
accepted, the script recomputes the centered second-moment exponent,
coefficient, R-squared, and transition count and requires numerical agreement
with every archived Stage-17 basin at rtol={VALIDATION_RTOL:g} and
atol={VALIDATION_ATOL:g}.

Within each starting-flow bin, the primary response is

`D2 = mean[(Delta q - mean(Delta q))^2] / (2 tau)`.

Two central-scale responses are fitted without deleting tail observations:

`D2_MAD = [median(|Delta q - median(Delta q)|) / 0.67448975]^2 / (2 tau)`

and

`D2_IQR = [(Q75(Delta q) - Q25(Delta q)) / 1.34897950]^2 / (2 tau)`.

The normal-consistency constants make the coefficients comparable under a
Gaussian distribution but do not affect slopes. Squaring each robust scale is
essential: an unsquared MAD or IQR slope estimates half of the
variance-equivalent exponent. Unweighted OLS of log response on log geometric
mean flow is used for all three estimators.

Continental medians and paired robust-minus-primary median differences use
{resamples:,} basin-bootstrap resamples with seed {seed}. Tail relationships
are marginal Spearman associations. They are descriptive rather than causal
because tail statistics were standardized with the primary variance law and
because basin attributes and forcing regimes covary. Benjamini-Hochberg values
control the false-discovery rate across the exported relationship table.
"""


def json_scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    started_utc = utc_now()
    started = time.perf_counter()
    args = parse_arguments()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.limit is not None and args.output is None:
        raise ValueError("--limit requires --output so a partial run cannot replace full outputs")

    cfg = load_config(args.config)
    out = (
        Path(args.output).expanduser().resolve()
        if args.output
        else output_root(cfg) / "sensitivity_analyses" / "robust_scale"
    )
    receipt_path = out / "_SUCCESS.json"
    if receipt_path.exists() and not args.force:
        raise RuntimeError(f"Outputs already complete at {out}; use --force to replace them")
    out.mkdir(parents=True, exist_ok=True)
    if args.force:
        receipt_path.unlink(missing_ok=True)

    stage10_root = stage_dir(cfg, 10, create=False)
    stage17_root = stage_dir(cfg, 17, create=False)
    stage10_receipt = stage10_root / "_SUCCESS.json"
    stage17_receipt = stage17_root / "_SUCCESS.json"
    stage17_path = stage17_root / "distribution_basin_diagnostics.csv"
    tail_root = output_root(cfg) / "sensitivity_analyses" / "tail_direction"
    tail_receipt = tail_root / "_SUCCESS.json"
    tail_path = tail_root / "basin_tail_direction_and_clustering.csv"
    required = [
        stage10_receipt,
        stage17_receipt,
        stage17_path,
        tail_receipt,
        tail_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")

    stage17_columns = [
        "GAGE_ID",
        "n_transitions",
        "variance_exponent",
        "variance_coefficient",
        "variance_r2",
        "standardized_excess_kurtosis",
        "quality_tier",
        "is_reference",
        "AGGECOREGION",
        "ecoregion",
        "aridity",
        "BFI_AVE",
        "log_area",
    ]
    stage17 = pd.read_csv(stage17_path, usecols=stage17_columns, dtype={"GAGE_ID": str})
    stage17["GAGE_ID"] = stage17["GAGE_ID"].astype(str).str.zfill(8)
    if stage17["GAGE_ID"].duplicated().any():
        raise ValueError("Stage-17 diagnostics contain duplicate GAGE_ID values")
    if args.limit:
        stage17 = stage17.sort_values("GAGE_ID").head(args.limit).reset_index(drop=True)

    tail_columns = [
        "GAGE_ID",
        "screen_variant",
        "two_sided_tail_multiple_of_normal",
        "positive_tail_multiple_of_normal",
        "negative_tail_multiple_of_normal",
        "jeffreys_tail_log_count_ratio",
    ]
    tail = pd.read_csv(tail_path, usecols=tail_columns, dtype={"GAGE_ID": str})
    tail["GAGE_ID"] = tail["GAGE_ID"].astype(str).str.zfill(8)
    tail = tail[tail["screen_variant"].eq("primary")].drop(columns="screen_variant")
    tail = tail[tail["GAGE_ID"].isin(stage17["GAGE_ID"])]
    if tail["GAGE_ID"].duplicated().any():
        raise ValueError("Primary directional-tail table contains duplicate GAGE_ID values")
    if len(tail) != len(stage17):
        raise ValueError(
            f"Directional-tail roster ({len(tail)}) differs from Stage 17 ({len(stage17)})"
        )

    settings = cfg["recession"]
    daily_dir = stage10_root / "daily"
    tasks = [
        (
            row.GAGE_ID,
            str(daily_dir / f"{row.GAGE_ID}.csv.gz"),
            int(row.n_transitions),
            int(settings["state_bins"]),
            int(settings["minimum_bin_count"]),
            int(settings["minimum_valid_bins_for_fit"]),
        )
        for row in stage17[["GAGE_ID", "n_transitions"]].itertuples(index=False)
    ]

    basin_rows: list[dict] = []
    bin_parts: list[pd.DataFrame] = []
    failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_basin, task): task[0] for task in tasks}
        for number, future in enumerate(as_completed(futures), start=1):
            gage = futures[future]
            try:
                row, bins, basin_failures = future.result()
            except Exception as error:
                row = {
                    "analysis_version": ANALYSIS_VERSION,
                    "GAGE_ID": gage,
                    "robust_scale_status": "unavailable",
                }
                bins = pd.DataFrame()
                basin_failures = [{
                    "GAGE_ID": gage,
                    "component": "worker",
                    "status": "unexpected_failure",
                    "reason": repr(error),
                }]
            basin_rows.append(row)
            if not bins.empty:
                bin_parts.append(bins)
            failures.extend(basin_failures)
            if number % 500 == 0 or number == len(tasks):
                elapsed = time.perf_counter() - started
                print(f"Completed {number:,}/{len(tasks):,} basins in {elapsed:.1f} s", flush=True)

    basin = pd.DataFrame(basin_rows).sort_values("GAGE_ID").reset_index(drop=True)
    bins = pd.concat(bin_parts, ignore_index=True).sort_values(
        ["GAGE_ID", "state_bin"]
    ).reset_index(drop=True)
    failures_frame = pd.DataFrame(
        failures,
        columns=["GAGE_ID", "component", "status", "reason"],
    ).sort_values(["GAGE_ID", "component"]).reset_index(drop=True)

    validation, validation_summary = validate_against_stage17(basin, stage17)
    validation_path = out / "validation_against_stage17.csv"
    validation_summary_path = out / "VALIDATION_SUMMARY.json"
    write_csv_atomic(validation, validation_path)
    write_json_atomic(validation_summary, validation_summary_path)
    if not validation_summary["all_metrics_match"]:
        raise RuntimeError(
            "Primary OLS reproduction failed; robust-scale summaries were not accepted"
        )

    frozen_attributes = stage17.drop(columns=[
        "n_transitions",
        "variance_exponent",
        "variance_coefficient",
        "variance_r2",
    ])
    basin = basin.merge(frozen_attributes, on="GAGE_ID", how="left", validate="one_to_one")
    basin = basin.merge(tail, on="GAGE_ID", how="left", validate="one_to_one")
    basin = basin.rename(columns={
        "ols_variance_exponent": "ols_variance_exponent",
        "mad2_variance_exponent": "mad2_variance_equivalent_exponent",
        "iqr2_variance_exponent": "iqr2_variance_equivalent_exponent",
        "mad2_variance_coefficient": "mad2_variance_equivalent_coefficient",
        "iqr2_variance_coefficient": "iqr2_variance_equivalent_coefficient",
        "mad2_variance_r2": "mad2_variance_equivalent_r2",
        "iqr2_variance_r2": "iqr2_variance_equivalent_r2",
    })
    basin["mad2_minus_ols"] = (
        basin["mad2_variance_equivalent_exponent"] - basin["ols_variance_exponent"]
    )
    basin["iqr2_minus_ols"] = (
        basin["iqr2_variance_equivalent_exponent"] - basin["ols_variance_exponent"]
    )
    basin["absolute_mad2_minus_ols"] = basin["mad2_minus_ols"].abs()
    basin["absolute_iqr2_minus_ols"] = basin["iqr2_minus_ols"].abs()
    basin["absolute_tail_log_count_ratio"] = basin["jeffreys_tail_log_count_ratio"].abs()

    population, paired = population_summaries(
        basin,
        args.bootstrap_resamples,
        args.seed,
    )
    relationships = tail_relationships(basin)
    quintiles = tail_quintile_summaries(basin)
    regions = ecoregion_summaries(basin)

    paths = {
        "basin": out / "basin_robust_scale_exponents.csv",
        "bins": out / "robust_scale_bin_statistics.csv.gz",
        "population": out / "robust_scale_population_summary.csv",
        "paired": out / "robust_scale_paired_comparisons.csv",
        "relationships": out / "robust_scale_tail_associations.csv",
        "quintiles": out / "robust_scale_tail_quintiles.csv",
        "regions": out / "robust_scale_ecoregion_summary.csv",
        "failures": out / "analysis_exclusions_and_failures.csv",
        "methods": out / "METHODS.md",
    }
    write_csv_atomic(basin, paths["basin"])
    write_csv_atomic(bins, paths["bins"], compression="gzip")
    write_csv_atomic(population, paths["population"])
    write_csv_atomic(paired, paths["paired"])
    write_csv_atomic(relationships, paths["relationships"])
    write_csv_atomic(quintiles, paths["quintiles"])
    write_csv_atomic(regions, paths["regions"])
    write_csv_atomic(failures_frame, paths["failures"])
    write_text_atomic(
        methods_text(settings, args.bootstrap_resamples, args.seed),
        paths["methods"],
    )

    elapsed_seconds = time.perf_counter() - started
    common_n = int(np.logical_and.reduce([
        np.isfinite(pd.to_numeric(basin[column], errors="coerce"))
        for _, _, _, column in ESTIMATORS
    ]).sum())
    pop_lookup = population.set_index("estimator_key")
    pair_lookup = paired.set_index("comparison_key")
    metrics = {
        "stage17_eligible_basins": int(len(stage17)),
        "primary_ols_reproduction_passed": bool(validation_summary["all_metrics_match"]),
        "maximum_ols_exponent_abs_difference": float(
            validation_summary["maximum_ols_variance_exponent_abs_difference"]
        ),
        "mad2_available_basins": int(basin["mad2_variance_equivalent_exponent"].notna().sum()),
        "iqr2_available_basins": int(basin["iqr2_variance_equivalent_exponent"].notna().sum()),
        "common_complete_case_basins": common_n,
        "primary_ols_median_exponent": float(pop_lookup.loc["ols", "median"]),
        "mad2_median_exponent": float(pop_lookup.loc["mad2", "median"]),
        "iqr2_median_exponent": float(pop_lookup.loc["iqr2", "median"]),
        "mad2_median_delta_vs_ols": float(
            pair_lookup.loc["mad2_minus_ols", "median_delta_robust_minus_primary"]
        ),
        "iqr2_median_delta_vs_ols": float(
            pair_lookup.loc["iqr2_minus_ols", "median_delta_robust_minus_primary"]
        ),
        "unexpected_failure_rows": int(failures_frame["status"].eq("unexpected_failure").sum()),
        "runtime_seconds": float(elapsed_seconds),
    }
    code_path = Path(__file__).resolve()
    output_paths = [*paths.values(), validation_path, validation_summary_path]
    receipt = {
        "analysis": "robust_scale_sensitivity",
        "analysis_version": ANALYSIS_VERSION,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "runtime_seconds": float(elapsed_seconds),
        "config": str(Path(cfg["_config_path"]).resolve()),
        "config_sha256": sha256_file(Path(cfg["_config_path"])),
        "code_sha256": {str(code_path.relative_to(Path(cfg["_project_root"]))): sha256_file(code_path)},
        "frozen_input_receipts_sha256": {
            str(stage10_receipt.relative_to(Path(cfg["_project_root"]))): sha256_file(stage10_receipt),
            str(stage17_receipt.relative_to(Path(cfg["_project_root"]))): sha256_file(stage17_receipt),
            str(tail_receipt.relative_to(Path(cfg["_project_root"]))): sha256_file(tail_receipt),
        },
        "frozen_input_tables_sha256": {
            str(stage17_path.relative_to(Path(cfg["_project_root"]))): sha256_file(stage17_path),
            str(tail_path.relative_to(Path(cfg["_project_root"]))): sha256_file(tail_path),
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "parameters": {
            "screen_column": PRIMARY_SCREEN_COLUMN,
            "lag_days": PRIMARY_LAG_DAYS,
            "state_bins": int(settings["state_bins"]),
            "minimum_bin_count": int(settings["minimum_bin_count"]),
            "minimum_valid_bins_for_fit": int(settings["minimum_valid_bins_for_fit"]),
            "mad_normalizer": MAD_NORMALIZER,
            "iqr_normalizer": IQR_NORMALIZER,
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "bootstrap_seed": int(args.seed),
            "workers": int(args.workers),
            "limit": int(args.limit) if args.limit else None,
        },
        "metrics": {key: json_scalar(value) for key, value in metrics.items()},
        "outputs": [str(path.resolve()) for path in output_paths],
        "output_sha256": {path.name: sha256_file(path) for path in output_paths},
    }
    write_json_atomic(receipt, receipt_path)
    print(f"Robust-scale sensitivity complete: {out}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
