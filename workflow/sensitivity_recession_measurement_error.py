#!/usr/bin/env python3
"""Sensitivity analysis for recession-exponent agreement and measurement error.

This script is intentionally outside the completed Stage 01--20 workflow.  It
reads frozen Stage 14, 15, and 17 tables and writes a self-contained set of
plot-ready source tables to::

    <configured-output-root>/sensitivity_analyses/mechanism_error/

It performs two analyses:

1. Direct agreement diagnostics for the hypothesis m = 2b under several
   matched definitions of the variance exponent m and drift exponent b.
2. A first-order multiplicative measurement-error envelope.  If

       Q_obs,t = Q_t (1 + epsilon_t),

   Var(epsilon_t) = sigma_e**2, and Corr(epsilon_t, epsilon_t+1) = rho_e,
   then the one-day error contribution to the finite-time variance rate is

       D2_error(q) = sigma_e**2 (1 - rho_e) q**2.

The measurement-error calculation is a scenario bound, not an estimate of
USGS measurement error and not an attribution of observed variance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from lib.common import DEFAULT_CONFIG, load_config, output_root, stage_dir


SCRIPT_VERSION = 1
DEFAULT_SEED = 20260907
DEFAULT_BOOTSTRAP_REPLICATES = 2000
FIT_R2_SCREEN = 0.50

CANONICAL_SIGMA_E = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15)
CANONICAL_RHO_E = (0.00, 0.25, 0.50, 0.75, 0.90, 0.95)
Q_REFERENCE = (0.25, 0.50, 1.00, 2.00, 4.00)
TARGET_RATIOS = (0.10, 0.25, 0.50, 1.00)


COMPARISONS = (
    {
        "comparison_id": "primary_m_vs_p_screened_monotone_b",
        "m_method": "p_screened_dry_state",
        "b_method": "p_screened_monotone",
        "same_transition_selection": False,
        "classical_recession_b": True,
        "interpretation": (
            "Primary all-sign, precipitation-screened dry-state m compared "
            "with b from precipitation-screened monotone declines."
        ),
        "selection_caution": (
            "The samples differ because the b fit conditions on negative "
            "increments; agreement or disagreement is not a same-estimand test."
        ),
    },
    {
        "comparison_id": "primary_m_vs_q_only_monotone_b",
        "m_method": "p_screened_dry_state",
        "b_method": "q_only_monotone",
        "same_transition_selection": False,
        "classical_recession_b": True,
        "interpretation": (
            "Primary all-sign, precipitation-screened dry-state m compared "
            "with b from discharge-only monotone declines."
        ),
        "selection_caution": (
            "Both forcing selection and sign selection differ between the two fits."
        ),
    },
    {
        "comparison_id": "same_p_screened_monotone_m_vs_2b",
        "m_method": "p_screened_monotone",
        "b_method": "p_screened_monotone",
        "same_transition_selection": True,
        "classical_recession_b": True,
        "interpretation": (
            "Variance and drift exponents estimated from the same "
            "precipitation-screened monotone-decline sample."
        ),
        "selection_caution": (
            "Conditioning on negative increments truncates the response used "
            "in both moments and can mechanically strengthen their association."
        ),
    },
    {
        "comparison_id": "same_q_only_monotone_m_vs_2b",
        "m_method": "q_only_monotone",
        "b_method": "q_only_monotone",
        "same_transition_selection": True,
        "classical_recession_b": True,
        "interpretation": (
            "Variance and drift exponents estimated from the same "
            "discharge-only monotone-decline sample."
        ),
        "selection_caution": (
            "The shared decline truncation can couple the two fitted moments, "
            "and the sample is not precipitation screened."
        ),
    },
    {
        "comparison_id": "same_primary_dry_state_m_vs_2b",
        "m_method": "p_screened_dry_state",
        "b_method": "p_screened_dry_state",
        "same_transition_selection": True,
        "classical_recession_b": False,
        "interpretation": (
            "Variance and drift exponents estimated from the same primary "
            "all-sign precipitation-screened dry-state sample."
        ),
        "selection_caution": (
            "Because positive increments are retained, this drift exponent is "
            "not a conventional recession exponent b."
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def normalize_gage(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.12g")
    temporary.replace(path)


def write_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def finite_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in columns:
        mask &= np.isfinite(pd.to_numeric(frame[column], errors="coerce"))
    return mask


def quantile_fields(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            f"{prefix}_p05": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p95": np.nan,
        }
    quantiles = np.quantile(values, (0.05, 0.25, 0.50, 0.75, 0.95))
    return dict(
        zip(
            (
                f"{prefix}_p05",
                f"{prefix}_p25",
                f"{prefix}_p50",
                f"{prefix}_p75",
                f"{prefix}_p95",
            ),
            map(float, quantiles),
        )
    )


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient for paired finite arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    denominator = float(x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    return float(2.0 * covariance / denominator) if denominator > 0 else np.nan


def direct_agreement_metrics(m: np.ndarray, two_b: np.ndarray) -> dict[str, float]:
    m = np.asarray(m, dtype=float)
    two_b = np.asarray(two_b, dtype=float)
    delta = m - two_b
    absolute = np.abs(delta)
    mean_pair = 0.5 * (m + two_b)
    if len(m) >= 2 and np.std(m) > 0 and np.std(two_b) > 0:
        pearson = float(stats.pearsonr(two_b, m).statistic)
        spearman = float(stats.spearmanr(two_b, m).statistic)
        ols_slope, ols_intercept = np.polyfit(two_b, m, 1)
    else:
        pearson = spearman = ols_slope = ols_intercept = np.nan
    delta_sd = float(np.std(delta, ddof=1)) if len(delta) > 1 else np.nan
    return {
        "n_basins": int(len(m)),
        "m_mean": float(np.mean(m)),
        "m_median": float(np.median(m)),
        "two_b_mean": float(np.mean(two_b)),
        "two_b_median": float(np.median(two_b)),
        "delta_mean": float(np.mean(delta)),
        "delta_median": float(np.median(delta)),
        "delta_sd": delta_sd,
        "delta_p05": float(np.quantile(delta, 0.05)),
        "delta_p95": float(np.quantile(delta, 0.95)),
        "median_absolute_delta": float(np.median(absolute)),
        "mean_absolute_delta": float(np.mean(absolute)),
        "root_mean_square_delta": float(np.sqrt(np.mean(delta**2))),
        "fraction_m_gt_two_b": float(np.mean(delta > 0)),
        "fraction_abs_delta_le_0p10": float(np.mean(absolute <= 0.10)),
        "fraction_abs_delta_le_0p25": float(np.mean(absolute <= 0.25)),
        "fraction_abs_delta_le_0p50": float(np.mean(absolute <= 0.50)),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "lin_concordance_correlation": concordance_correlation(two_b, m),
        "descriptive_ols_intercept": float(ols_intercept),
        "descriptive_ols_slope": float(ols_slope),
        "bland_altman_mean_axis_median": float(np.median(mean_pair)),
        "bland_altman_lower_95": float(np.mean(delta) - 1.96 * delta_sd),
        "bland_altman_upper_95": float(np.mean(delta) + 1.96 * delta_sd),
    }


def bootstrap_direct_metrics(
    frame: pd.DataFrame,
    replicate_count: int,
    rng: np.random.Generator,
    scheme: str,
) -> dict[str, tuple[float, float]]:
    """Bootstrap direct equality metrics, optionally resampling ecoregions."""
    metrics = {
        "m_median": [],
        "two_b_median": [],
        "delta_median": [],
        "median_absolute_delta": [],
        "fraction_abs_delta_le_0p25": [],
    }
    m = frame["m"].to_numpy(dtype=float)
    two_b = frame["two_b"].to_numpy(dtype=float)
    n = len(frame)
    if replicate_count <= 0 or n == 0:
        return {name: (np.nan, np.nan) for name in metrics}

    if scheme == "basin_iid":
        def sample_indices() -> np.ndarray:
            return rng.integers(0, n, size=n)
    elif scheme == "macro_ecoregion_cluster":
        # AGGECOREGION is the nine-class GAGES-II aggregated ecoregion.  The
        # separate Stage-15 `ecoregion` field is a plotting subgroup that labels
        # all reference basins "Reference" and therefore is not a spatial block.
        groups = frame["AGGECOREGION"].fillna("MISSING").astype(str).to_numpy()
        unique = np.unique(groups)
        group_indices = [np.flatnonzero(groups == group) for group in unique]

        def sample_indices() -> np.ndarray:
            draws = rng.integers(0, len(group_indices), size=len(group_indices))
            return np.concatenate([group_indices[index] for index in draws])
    else:
        raise ValueError(f"Unknown bootstrap scheme: {scheme}")

    for _ in range(replicate_count):
        indices = sample_indices()
        sampled_m = m[indices]
        sampled_two_b = two_b[indices]
        delta = sampled_m - sampled_two_b
        metrics["m_median"].append(float(np.median(sampled_m)))
        metrics["two_b_median"].append(float(np.median(sampled_two_b)))
        metrics["delta_median"].append(float(np.median(delta)))
        metrics["median_absolute_delta"].append(float(np.median(np.abs(delta))))
        metrics["fraction_abs_delta_le_0p25"].append(
            float(np.mean(np.abs(delta) <= 0.25))
        )
    return {
        name: tuple(map(float, np.quantile(values, (0.025, 0.975))))
        for name, values in metrics.items()
    }


def prepare_attributes(stage15_path: Path) -> pd.DataFrame:
    wanted = [
        "GAGE_ID",
        "LAT_GAGE",
        "LNG_GAGE",
        "STATE",
        "ecoregion",
        "AGGECOREGION",
        "is_reference",
        "quality_tier",
        "area_km2",
        "aridity",
        "BFI_AVE",
        "log_area",
    ]
    header = pd.read_csv(stage15_path, nrows=0).columns
    columns = [column for column in wanted if column in header]
    attributes = pd.read_csv(
        stage15_path, usecols=columns, dtype={"GAGE_ID": str}, low_memory=False
    )
    attributes["GAGE_ID"] = normalize_gage(attributes["GAGE_ID"])
    if attributes["GAGE_ID"].duplicated().any():
        raise ValueError("Stage 15 contains duplicate GAGE_ID values")
    return attributes


def analyze_m_vs_2b(
    recession_path: Path,
    attributes: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recession = pd.read_csv(recession_path, dtype={"GAGE_ID": str})
    recession["GAGE_ID"] = normalize_gage(recession["GAGE_ID"])
    recession = recession.loc[pd.to_numeric(recession["tau_days"], errors="coerce").eq(1)].copy()
    if recession.duplicated(["GAGE_ID", "method"]).any():
        raise ValueError("Stage 14 has duplicate basin/method rows at tau=1")

    pair_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    interval_rows: list[dict] = []
    definition_rows: list[dict] = []

    for comparison_number, definition in enumerate(COMPARISONS):
        m_method = recession.loc[
            recession["method"].eq(definition["m_method"]),
            [
                "GAGE_ID",
                "variance_exponent",
                "variance_coefficient",
                "variance_r2",
                "variance_bins",
                "n_transitions",
            ],
        ].rename(
            columns={
                "variance_exponent": "m",
                "variance_coefficient": "m_coefficient",
                "variance_r2": "m_r2",
                "variance_bins": "m_bins",
                "n_transitions": "m_n_transitions",
            }
        )
        b_method = recession.loc[
            recession["method"].eq(definition["b_method"]),
            [
                "GAGE_ID",
                "drift_exponent",
                "drift_coefficient",
                "drift_r2",
                "n_transitions",
            ],
        ].rename(
            columns={
                "drift_exponent": "b",
                "drift_coefficient": "b_coefficient",
                "drift_r2": "b_r2",
                "n_transitions": "b_n_transitions",
            }
        )
        pairs = m_method.merge(b_method, on="GAGE_ID", how="inner", validate="one_to_one")
        pairs = pairs.loc[finite_numeric(pairs, ("m", "b", "m_r2", "b_r2"))].copy()
        pairs["two_b"] = 2.0 * pairs["b"]
        pairs["delta_m_minus_2b"] = pairs["m"] - pairs["two_b"]
        pairs["mean_m_and_2b"] = 0.5 * (pairs["m"] + pairs["two_b"])
        pairs["absolute_delta"] = pairs["delta_m_minus_2b"].abs()
        pairs["ratio_m_to_2b"] = np.where(
            np.abs(pairs["two_b"]) > 1e-12, pairs["m"] / pairs["two_b"], np.nan
        )
        pairs["b_positive"] = pairs["b"] > 0
        pairs["both_r2_ge_0p5"] = (pairs["m_r2"] >= FIT_R2_SCREEN) & (
            pairs["b_r2"] >= FIT_R2_SCREEN
        )
        pairs["fit_screen"] = pairs["b_positive"] & pairs["both_r2_ge_0p5"]
        pairs.insert(0, "comparison_id", definition["comparison_id"])
        pairs = pairs.merge(attributes, on="GAGE_ID", how="left", validate="many_to_one")
        pair_frames.append(pairs)

        definition_rows.append(
            {
                **definition,
                "all_finite_n": int(len(pairs)),
                "fit_screen_n": int(pairs["fit_screen"].sum()),
                "fit_screen_definition": (
                    f"m_r2 >= {FIT_R2_SCREEN:.2f}, b_r2 >= {FIT_R2_SCREEN:.2f}, and b > 0; "
                    "exploratory, not prespecified"
                ),
            }
        )

        subsets = (
            ("all_finite", pairs),
            ("exploratory_fit_screen", pairs.loc[pairs["fit_screen"]].copy()),
        )
        for subset_number, (subset_name, subset) in enumerate(subsets):
            metrics = direct_agreement_metrics(
                subset["m"].to_numpy(float), subset["two_b"].to_numpy(float)
            )
            summary_rows.append(
                {
                    "comparison_id": definition["comparison_id"],
                    "subset": subset_name,
                    "m_method": definition["m_method"],
                    "b_method": definition["b_method"],
                    "same_transition_selection": definition["same_transition_selection"],
                    "classical_recession_b": definition["classical_recession_b"],
                    **metrics,
                }
            )
            for scheme_number, scheme in enumerate(("basin_iid", "macro_ecoregion_cluster")):
                rng = np.random.default_rng(
                    seed
                    + 10_000 * comparison_number
                    + 1_000 * subset_number
                    + 100 * scheme_number
                )
                intervals = bootstrap_direct_metrics(
                    subset, bootstrap_replicates, rng, scheme
                )
                for metric, (low, high) in intervals.items():
                    interval_rows.append(
                        {
                            "comparison_id": definition["comparison_id"],
                            "subset": subset_name,
                            "bootstrap_scheme": scheme,
                            "bootstrap_replicates": bootstrap_replicates,
                            "metric": metric,
                            "ci_low_2p5": low,
                            "ci_high_97p5": high,
                        }
                    )

    return (
        pd.concat(pair_frames, ignore_index=True),
        pd.DataFrame(summary_rows),
        pd.DataFrame(interval_rows),
        pd.DataFrame(definition_rows),
    )


def measurement_ratio(
    variance_coefficient: np.ndarray,
    variance_exponent: np.ndarray,
    sigma_e: float,
    rho_e: float,
    q_norm: float,
) -> np.ndarray:
    error_coefficient = sigma_e**2 * (1.0 - rho_e)
    return (
        error_coefficient
        * np.power(q_norm, 2.0 - variance_exponent)
        / variance_coefficient
    )


def summarize_measurement_ratio(
    ratios: np.ndarray,
    sigma_e: float,
    rho_e: float,
    q_norm: float,
) -> dict:
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    result = {
        "sigma_e": float(sigma_e),
        "rho_e": float(rho_e),
        "effective_unshared_error_sd": float(sigma_e * np.sqrt(1.0 - rho_e)),
        "d2_error_coefficient": float(sigma_e**2 * (1.0 - rho_e)),
        "q_norm": float(q_norm),
        "n_basins": int(len(ratios)),
        **quantile_fields(ratios, "error_to_observed_d2_ratio"),
        "mean_capped_error_fraction": float(np.mean(np.minimum(ratios, 1.0))),
    }
    for threshold in TARGET_RATIOS:
        label = str(threshold).replace(".", "p")
        result[f"fraction_ratio_ge_{label}"] = float(np.mean(ratios >= threshold))
    return result


def analyze_measurement_error(
    distribution_path: Path,
    recession_path: Path,
    attributes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    diagnostics = pd.read_csv(distribution_path, dtype={"GAGE_ID": str})
    diagnostics["GAGE_ID"] = normalize_gage(diagnostics["GAGE_ID"])
    keep = ["GAGE_ID", "variance_coefficient", "variance_exponent", "variance_r2"]
    diagnostics = diagnostics[keep].copy()
    diagnostics = diagnostics.loc[
        finite_numeric(diagnostics, ("variance_coefficient", "variance_exponent", "variance_r2"))
        & diagnostics["variance_coefficient"].gt(0)
    ].copy()
    if diagnostics["GAGE_ID"].duplicated().any():
        raise ValueError("Stage 17 distribution diagnostics contain duplicate GAGE_ID values")

    # Verify that Stage 17's fitted law is exactly the frozen primary Stage 14 law.
    recession = pd.read_csv(recession_path, dtype={"GAGE_ID": str})
    recession["GAGE_ID"] = normalize_gage(recession["GAGE_ID"])
    primary = recession.loc[
        recession["method"].eq("p_screened_dry_state")
        & pd.to_numeric(recession["tau_days"], errors="coerce").eq(1),
        ["GAGE_ID", "variance_coefficient", "variance_exponent", "variance_r2"],
    ]
    checked = diagnostics.merge(
        primary, on="GAGE_ID", how="left", suffixes=("_stage17", "_stage14"), validate="one_to_one"
    )
    differences = {}
    for column in ("variance_coefficient", "variance_exponent", "variance_r2"):
        difference = np.abs(
            checked[f"{column}_stage17"] - checked[f"{column}_stage14"]
        )
        differences[f"maximum_abs_{column}_difference"] = float(difference.max())
    if any(value > 1e-12 for value in differences.values()):
        raise ValueError("Stage 14 and Stage 17 primary variance fits are inconsistent")

    basin_reference = diagnostics.merge(
        attributes, on="GAGE_ID", how="left", validate="one_to_one"
    )
    for target in TARGET_RATIOS:
        label = str(target).replace(".", "p")
        basin_reference[f"critical_effective_error_sd_for_ratio_{label}_q1"] = np.sqrt(
            target * basin_reference["variance_coefficient"]
        )

    coefficients = basin_reference["variance_coefficient"].to_numpy(float)
    exponents = basin_reference["variance_exponent"].to_numpy(float)
    basin_scenario_frames: list[pd.DataFrame] = []
    scenario_summary_rows: list[dict] = []
    q_summary_rows: list[dict] = []
    for sigma_e in CANONICAL_SIGMA_E:
        for rho_e in CANONICAL_RHO_E:
            scenario_id = (
                f"sigma_{sigma_e:.3f}_rho_{rho_e:.2f}".replace(".", "p")
            )
            ratio_q1 = measurement_ratio(coefficients, exponents, sigma_e, rho_e, 1.0)
            scenario = pd.DataFrame(
                {
                    "scenario_id": scenario_id,
                    "GAGE_ID": basin_reference["GAGE_ID"],
                    "sigma_e": sigma_e,
                    "rho_e": rho_e,
                    "effective_unshared_error_sd": sigma_e * np.sqrt(1.0 - rho_e),
                    "d2_error_coefficient": sigma_e**2 * (1.0 - rho_e),
                    "observed_d2_coefficient": coefficients,
                    "observed_variance_exponent": exponents,
                    "observed_variance_r2": basin_reference["variance_r2"].to_numpy(float),
                    "error_to_observed_d2_ratio_q1": ratio_q1,
                    "capped_error_fraction_q1": np.minimum(ratio_q1, 1.0),
                }
            )
            basin_scenario_frames.append(scenario)
            scenario_summary_rows.append(
                {
                    "scenario_id": scenario_id,
                    **summarize_measurement_ratio(ratio_q1, sigma_e, rho_e, 1.0),
                }
            )
            for q_norm in Q_REFERENCE:
                ratio = measurement_ratio(
                    coefficients, exponents, sigma_e, rho_e, q_norm
                )
                q_summary_rows.append(
                    {
                        "scenario_id": scenario_id,
                        **summarize_measurement_ratio(
                            ratio, sigma_e, rho_e, q_norm
                        ),
                    }
                )

    # Dense q=1 surface for a contour/heat-map panel.  It is a deterministic
    # transform of the same equation, not an additional set of assumptions.
    dense_sigma = np.round(np.arange(0.0, 0.150001, 0.005), 3)
    dense_rho = np.unique(
        np.concatenate((np.round(np.arange(0.0, 0.950001, 0.05), 2), [0.99]))
    )
    surface_rows = []
    for sigma_e in dense_sigma:
        for rho_e in dense_rho:
            ratio = measurement_ratio(coefficients, exponents, sigma_e, rho_e, 1.0)
            surface_rows.append(
                summarize_measurement_ratio(ratio, sigma_e, rho_e, 1.0)
            )

    critical_rows = []
    for target in TARGET_RATIOS:
        effective_sigma = np.sqrt(target * coefficients)
        for rho_e in CANONICAL_RHO_E:
            required_sigma = effective_sigma / np.sqrt(1.0 - rho_e)
            critical_rows.append(
                {
                    "target_error_to_observed_d2_ratio_q1": target,
                    "assumed_rho_e": rho_e,
                    "n_basins": len(required_sigma),
                    **quantile_fields(required_sigma, "required_sigma_e"),
                }
            )

    observed_summary = {
        "n_basins": int(len(basin_reference)),
        **quantile_fields(coefficients, "observed_d2_coefficient"),
        **quantile_fields(np.sqrt(2.0 * coefficients), "observed_increment_sd_at_q1"),
        **differences,
    }
    return (
        basin_reference,
        pd.concat(basin_scenario_frames, ignore_index=True),
        pd.DataFrame(scenario_summary_rows),
        pd.DataFrame(q_summary_rows),
        pd.DataFrame(surface_rows),
        {"observed": observed_summary, "critical_rows": critical_rows},
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    cfg = load_config(args.config)
    repo_root = Path(cfg["_project_root"])
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else output_root(cfg) / "sensitivity_analyses" / "mechanism_error"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    recession_path = stage_dir(cfg, 14, create=False) / "recession_basin_method_summary.csv"
    stage15_path = stage_dir(cfg, 15, create=False) / "continental_basin_results.csv"
    distribution_path = stage_dir(cfg, 17, create=False) / "distribution_basin_diagnostics.csv"
    input_paths = (recession_path, stage15_path, distribution_path)
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")

    attributes = prepare_attributes(stage15_path)
    pairs, m_summary, bootstrap_intervals, definitions = analyze_m_vs_2b(
        recession_path, attributes, args.bootstrap_replicates, args.seed
    )
    (
        measurement_reference,
        measurement_basin_scenarios,
        measurement_scenarios,
        measurement_q_summary,
        measurement_surface,
        measurement_metadata,
    ) = analyze_measurement_error(distribution_path, recession_path, attributes)
    critical_summary = pd.DataFrame(measurement_metadata.pop("critical_rows"))

    output_frames = {
        "m_2b_comparison_definitions.csv": definitions,
        "m_2b_basin_pairs.csv": pairs,
        "m_2b_comparison_summary.csv": m_summary,
        "m_2b_bootstrap_intervals.csv": bootstrap_intervals,
        "measurement_error_basin_reference.csv": measurement_reference,
        "measurement_error_basin_scenarios.csv": measurement_basin_scenarios,
        "measurement_error_scenario_summary.csv": measurement_scenarios,
        "measurement_error_q_sensitivity_summary.csv": measurement_q_summary,
        "measurement_error_surface_summary.csv": measurement_surface,
        "measurement_error_critical_sigma_summary.csv": critical_summary,
    }
    output_paths = []
    for filename, frame in output_frames.items():
        path = output_dir / filename
        write_csv(frame, path)
        output_paths.append(path)

    primary_comparisons = m_summary.loc[
        m_summary["subset"].eq("all_finite")
        & m_summary["comparison_id"].isin(
            (
                "primary_m_vs_p_screened_monotone_b",
                "primary_m_vs_q_only_monotone_b",
            )
        )
    ]
    canonical_highlights = measurement_scenarios.loc[
        measurement_scenarios["sigma_e"].isin((0.03, 0.05))
        & measurement_scenarios["rho_e"].isin((0.0, 0.9))
    ]
    analysis_summary = {
        "script_version": SCRIPT_VERSION,
        "scope": (
            "Supplementary sensitivity analysis using frozen Stage 14, 15, and 17 outputs; "
            "no Stage 01-20 output was modified."
        ),
        "m_equals_2b": {
            "hypothesis": "m = 2b",
            "primary_comparisons": primary_comparisons.to_dict(orient="records"),
            "regression_choice": (
                "Deming/orthogonal regression was not used because the relative error "
                "variance of the two fitted exponents is not known. Descriptive OLS is "
                "reported only as plot geometry. Paired differences and agreement are primary."
            ),
            "bootstrap": {
                "replicates": args.bootstrap_replicates,
                "seed": args.seed,
                "schemes": [
                    "iid basin resampling",
                    "macro-ecoregion cluster resampling",
                ],
                "caution": (
                    "The nine-region cluster bootstrap is a coarse spatial sensitivity, "
                    "not a complete model of spatial dependence."
                ),
            },
        },
        "measurement_error": {
            "observation_model": "Q_obs,t = Q_t * (1 + epsilon_t)",
            "finite_time_rate_equation": (
                "D2_error(q) = sigma_e^2 * (1 - rho_e) * q^2"
            ),
            "comparison_equation": (
                "R_error(q) = [sigma_e^2 * (1-rho_e) / A_obs] * q^(2-m_obs)"
            ),
            "canonical_highlights_q1": canonical_highlights.to_dict(orient="records"),
            **measurement_metadata,
            "cautions": [
                "sigma_e and rho_e are scenarios, not inferred site-specific errors.",
                "The approximation is first order in relative error.",
                "Ratios compare fitted variance rates at normalized flow q and do not identify causation.",
                "A ratio above one means the scenario is incompatible with treating all assumed error as an additive component of that basin's fitted D2 at that q.",
                "The q=1 comparison is the coefficient comparison; other q values depend on the observed exponent and may extend beyond an individual basin's fitted flow range.",
            ],
        },
    }
    summary_path = output_dir / "analysis_summary.json"
    write_json(analysis_summary, summary_path)
    output_paths.append(summary_path)

    # Basic deterministic checks protect the plot tables from silent algebra or
    # join errors without rerunning any upstream analysis.
    if len(measurement_reference) != 3727:
        raise AssertionError(
            f"Expected 3,727 Stage-17 basins, found {len(measurement_reference)}"
        )
    check = measurement_scenarios.loc[
        measurement_scenarios["sigma_e"].eq(0.03)
        & measurement_scenarios["rho_e"].eq(0.0)
    ].iloc[0]
    expected_median = 0.03**2 / float(
        measurement_reference["variance_coefficient"].median()
    )
    if not np.isclose(
        check["error_to_observed_d2_ratio_p50"], expected_median, rtol=1e-12
    ):
        raise AssertionError("Measurement-error median ratio check failed")
    required_comparisons = {definition["comparison_id"] for definition in COMPARISONS}
    if set(pairs["comparison_id"].unique()) != required_comparisons:
        raise AssertionError("One or more m-versus-2b comparisons are missing")

    receipt = {
        "script_version": SCRIPT_VERSION,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": ".",
        "output_directory": str(output_dir.relative_to(repo_root)),
        "random_seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "inputs": {
            str(path.relative_to(repo_root)): sha256_file(path)
            for path in input_paths
        },
        "code": {
            str(Path(__file__).resolve().relative_to(repo_root)): sha256_file(
                Path(__file__).resolve()
            )
        },
        "outputs": {
            str(path.relative_to(repo_root)): sha256_file(path)
            for path in output_paths
        },
        "metrics": {
            "stage17_basins": int(len(measurement_reference)),
            "m_2b_comparisons": int(definitions["comparison_id"].nunique()),
            "m_2b_pair_rows": int(len(pairs)),
            "canonical_measurement_scenarios": int(len(measurement_scenarios)),
            "dense_measurement_surface_scenarios": int(len(measurement_surface)),
        },
        "upstream_outputs_modified": False,
        "self_checks_passed": True,
    }
    receipt_path = output_dir / "_SUCCESS.json"
    write_json(receipt, receipt_path)

    print(f"Wrote sensitivity-analysis outputs to {output_dir}")
    print(
        m_summary.loc[m_summary["subset"].eq("all_finite"), [
            "comparison_id", "n_basins", "m_median", "two_b_median",
            "delta_median", "fraction_abs_delta_le_0p25", "spearman_rho",
        ]].to_string(index=False)
    )
    print("\nCanonical measurement-error scenarios at q=1:")
    print(
        canonical_highlights[[
            "sigma_e", "rho_e", "error_to_observed_d2_ratio_p50",
            "fraction_ratio_ge_0p25", "fraction_ratio_ge_1p0",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
