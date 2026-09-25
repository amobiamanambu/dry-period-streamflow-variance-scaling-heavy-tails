#!/usr/bin/env python3
"""Stage 18: test heavy-tail and lead-adaptive conditional low-flow probabilities."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib.common import (
    atomic_target, configure_logging, load_config, parse_args, prepare_stage,
    require_stage, stage_dir, write_receipt,
)
from lib.hydrology import contiguous_run_filter
from lib.statistics import spearman_table


STAGE = 18
ANALYSIS_VERSION = 8
MAIN_MODEL = "regularized_power_state_empirical"
GAUSSIAN_MODEL = "regularized_power_gaussian"
STATE_GAUSSIAN_MODEL = "regularized_power_state_gaussian"
OPERATIONAL_MODEL = "adaptive_heavytail_recession_logistic"
BASELINE_MODELS = ["training_climatology", "persistence", "qbin_probability"]
ALL_MODELS = [
    *BASELINE_MODELS, GAUSSIAN_MODEL, STATE_GAUSSIAN_MODEL, MAIN_MODEL,
    OPERATIONAL_MODEL,
]
EXCLUSION_COLUMNS = ["GAGE_ID", "lead_days", "threshold_name", "reason"]
SEVERITY_COLUMNS = [
    "extreme_threshold_quantile", "extreme_threshold_name",
    "comparison_threshold_quantile", "comparison_threshold_name",
    "reference_model", "lead_days", "basins", "spatial_groups", "water_years",
    "weighted_q5_minus_comparison_brier_skill", "bootstrap_ci_low",
    "bootstrap_ci_high", "positive_means_q5_retains_more_skill",
]


def threshold_name(quantile: float) -> str:
    return f"Q{int(round(100 * quantile))}"


def dry_transition_frame(frame: pd.DataFrame, lead: int, cfg: dict) -> pd.DataFrame:
    """Return direct-lead transitions inside observed no-effective-input periods."""
    work = frame.sort_values("date").copy()
    setting = cfg["transition_definitions"]
    cutoff = float(setting["precipitation_thresholds_mm_day"][0])
    antecedent = 3 if 3 in setting["antecedent_windows_days"] else int(
        setting["antecedent_windows_days"][0]
    )
    dates = pd.to_datetime(work["date"], errors="coerce")
    q = pd.to_numeric(work["q_norm_predictability"], errors="coerce")
    future = q.shift(-lead)
    consecutive = (dates.shift(-lead) - dates).dt.days.eq(lead)
    valid = consecutive & q.gt(0) & future.ge(0)

    offsets = range(-(antecedent - 1), lead + 1)
    precipitation = pd.concat(
        [pd.to_numeric(work["pr_mm"], errors="coerce").shift(-offset) for offset in offsets],
        axis=1,
    ).max(axis=1, skipna=False)
    melt = work["snowmelt_risk"].astype("boolean")
    melt_components = pd.concat([melt.shift(-offset) for offset in offsets], axis=1)
    possible_melt_or_missing = (
        melt_components.eq(True).any(axis=1) | melt_components.isna().any(axis=1)
    )
    dry = precipitation.le(cutoff) & ~possible_melt_or_missing
    selected = contiguous_run_filter(valid & dry, int(setting["minimum_contiguous_days"]))
    return pd.DataFrame({
        "date": dates[selected],
        "future_date": dates.shift(-lead)[selected],
        "q": q[selected].to_numpy(float),
        "q_next": future[selected].to_numpy(float),
        "dq": (future - q)[selected].to_numpy(float),
    })


def dry_spell_age(dry: pd.Series) -> pd.Series:
    """Count consecutive no-effective-input days known at forecast initialization."""
    values = dry.fillna(False).to_numpy(bool)
    age = np.zeros(len(values), dtype=float)
    current = 0
    for index, value in enumerate(values):
        current = current + 1 if value else 0
        age[index] = current
    return pd.Series(age, index=dry.index)


def initialization_features(
    frame: pd.DataFrame, discharge_scale: float, cfg: dict
) -> tuple[pd.DataFrame, list[str]]:
    """Build hydrograph-memory predictors available on the initialization date."""
    work = frame.sort_values("date").copy()
    q = pd.to_numeric(work["q_mm_day"], errors="coerce") / discharge_scale
    log_q = np.log(np.maximum(q, 1e-10))
    precipitation = pd.to_numeric(work["pr_mm"], errors="coerce")
    melt = work["snowmelt_risk"].astype("boolean")
    cutoff = float(cfg["transition_definitions"]["precipitation_thresholds_mm_day"][0])
    current_dry = precipitation.le(cutoff) & melt.eq(False)
    dates = pd.to_datetime(work["date"], errors="coerce")
    features = pd.DataFrame({
        "date": dates,
        "log_q": log_q,
        "dlog_q_1": log_q - log_q.shift(1),
        "dlog_q_3": (log_q - log_q.shift(3)) / 3.0,
        "dlog_q_7": (log_q - log_q.shift(7)) / 7.0,
        "dlog_q_14": (log_q - log_q.shift(14)) / 14.0,
        "log_q_minus_mean_7": log_q - log_q.rolling(7, min_periods=4).mean(),
        "log_q_volatility_7": log_q.diff().rolling(7, min_periods=4).std(),
        "log_q_volatility_14": log_q.diff().rolling(14, min_periods=7).std(),
        "dry_spell_age": dry_spell_age(current_dry),
        "doy_sin": np.sin(2 * np.pi * dates.dt.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * dates.dt.dayofyear / 365.25),
    })
    columns = [column for column in features.columns if column != "date"]
    return features, columns


def recession_logistic_probabilities(
    train: pd.DataFrame, test: pd.DataFrame, target: np.ndarray,
    feature_columns: list[str], cfg: dict,
) -> np.ndarray:
    """Estimate a threshold-specific probability from antecedent hydrograph memory."""
    setting = cfg["predictability_test"]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=float(setting["recession_logistic_regularization_c"]),
            max_iter=int(setting["recession_logistic_max_iterations"]),
            solver="lbfgs",
        ),
    )
    model.fit(train[feature_columns], np.asarray(target, dtype=bool))
    return model.predict_proba(test[feature_columns])[:, 1]


def fit_transition_model(train: pd.DataFrame, cfg: dict) -> dict:
    """Fit a regularized direct-lead conditional location-scale model."""
    setting = cfg["predictability_test"]
    q = train["q"].to_numpy(float)
    dq = train["dq"].to_numpy(float)
    edges = np.unique(np.quantile(q, np.linspace(0, 1, int(setting["state_bins"]) + 1)))
    if len(edges) < int(setting["minimum_valid_bins_for_fit"]) + 1:
        raise ValueError("too_few_unique_flow_bins")
    assignment = np.searchsorted(edges[1:-1], q, side="right")
    rows = []
    for state in range(len(edges) - 1):
        selected = assignment == state
        if selected.sum() < int(setting["minimum_bin_count"]):
            continue
        increments = dq[selected]
        variance = float(np.var(increments, ddof=1))
        if not np.isfinite(variance) or variance <= 0:
            continue
        rows.append({
            "q_center": float(np.exp(np.mean(np.log(q[selected])))),
            "mean_increment": float(np.mean(increments)),
            "variance": variance,
            "n": int(selected.sum()),
        })
    bins = pd.DataFrame(rows)
    if len(bins) < int(setting["minimum_valid_bins_for_fit"]):
        raise ValueError(f"only_{len(bins)}_valid_conditional_bins")

    fit = stats.linregress(np.log(bins["q_center"]), np.log(bins["variance"]))
    raw_exponent = float(fit.slope)
    exponent_se = float(fit.stderr) if fit.stderr is not None else np.nan
    prior_mean = float(setting["variance_exponent_prior_mean"])
    prior_sd = float(setting["variance_exponent_prior_sd"])
    if np.isfinite(exponent_se) and exponent_se > 0:
        raw_precision = 1.0 / exponent_se**2
        prior_precision = 1.0 / prior_sd**2
        regularized_exponent = (
            raw_exponent * raw_precision + prior_mean * prior_precision
        ) / (raw_precision + prior_precision)
    elif np.isfinite(raw_exponent):
        regularized_exponent = raw_exponent
    else:
        regularized_exponent = prior_mean
    low, high = [float(item) for item in setting["variance_exponent_bounds"]]
    regularized_exponent = float(np.clip(regularized_exponent, low, high))
    coefficient = float(np.exp(np.average(
        np.log(bins["variance"]) - regularized_exponent * np.log(bins["q_center"]),
        weights=bins["n"],
    )))

    order = np.argsort(bins["q_center"].to_numpy(float))
    centers = bins["q_center"].to_numpy(float)[order]
    means = bins["mean_increment"].to_numpy(float)[order]
    q_low, q_high = float(centers[0]), float(centers[-1])
    q_model = np.clip(q, q_low, q_high)
    conditional_mean = np.interp(
        np.log(q_model), np.log(centers), means, left=means[0], right=means[-1]
    )
    base_sigma = np.sqrt(coefficient * np.power(q_model, regularized_exponent))
    standardized = (dq - conditional_mean) / base_sigma
    finite = np.isfinite(standardized)
    if finite.sum() < int(setting["minimum_training_transitions"]):
        raise ValueError("insufficient_finite_standardized_residuals")
    q_finite = q[finite]
    standardized = standardized[finite]
    global_mean = float(np.mean(standardized))
    global_sd = float(np.std(standardized, ddof=1))
    if not np.isfinite(global_sd) or global_sd <= 0:
        raise ValueError("standardized_residual_scale_unavailable")

    group_edges = np.unique(np.quantile(
        q_finite,
        np.linspace(0, 1, int(setting["residual_flow_groups"]) + 1),
    ))
    if len(group_edges) < 3:
        raise ValueError("too_few_residual_flow_groups")
    group_assignment = np.searchsorted(group_edges[1:-1], q_finite, side="right")
    residual_groups = {}
    residual_group_moments = {}
    for group in range(len(group_edges) - 1):
        values = standardized[group_assignment == group]
        if len(values) >= int(setting["minimum_bin_count"]):
            residual_groups[group] = np.sort(values)
            residual_group_moments[group] = (
                float(np.mean(values)), float(np.std(values, ddof=1))
            )
    if len(residual_groups) < 2:
        raise ValueError("too_few_empirical_residual_groups")

    return {
        "q_centers": centers,
        "mean_increments": means,
        "q_model_low": q_low,
        "q_model_high": q_high,
        "bin_edges": edges,
        "residual_group_edges": group_edges,
        "residual_groups": residual_groups,
        "residual_group_moments": residual_group_moments,
        "global_empirical_standardized": np.sort(standardized),
        "global_standardized_mean": global_mean,
        "global_standardized_sd": global_sd,
        "raw_variance_exponent": raw_exponent,
        "variance_exponent_standard_error": exponent_se,
        "regularized_variance_exponent": regularized_exponent,
        "variance_coefficient": coefficient,
        "variance_r2": float(fit.rvalue**2),
    }


def model_components(q: np.ndarray, fitted: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_model = np.clip(q, fitted["q_model_low"], fitted["q_model_high"])
    mean_increment = np.interp(
        np.log(q_model), np.log(fitted["q_centers"]), fitted["mean_increments"],
        left=fitted["mean_increments"][0], right=fitted["mean_increments"][-1],
    )
    sigma = np.sqrt(
        fitted["variance_coefficient"]
        * np.power(q_model, fitted["regularized_variance_exponent"])
    )
    return q_model, mean_increment, sigma


def distribution_members(model: str, q: np.ndarray, fitted: dict, members: int) -> np.ndarray:
    probabilities = (np.arange(members, dtype=float) + 0.5) / members
    q_model, mean_increment, sigma = model_components(q, fitted)
    if model == GAUSSIAN_MODEL:
        location = q + mean_increment + sigma * fitted["global_standardized_mean"]
        spread = sigma * fitted["global_standardized_sd"]
        ensemble = location[:, None] + spread[:, None] * stats.norm.ppf(probabilities)[None, :]
    elif model in {STATE_GAUSSIAN_MODEL, MAIN_MODEL}:
        assignments = np.searchsorted(
            fitted["residual_group_edges"][1:-1], q_model, side="right"
        )
        ensemble = np.empty((len(q), members), dtype=float)
        fallback = fitted["global_empirical_standardized"]
        for group in np.unique(assignments):
            selected = assignments == group
            if model == STATE_GAUSSIAN_MODEL:
                center, spread = fitted["residual_group_moments"].get(
                    int(group),
                    (fitted["global_standardized_mean"], fitted["global_standardized_sd"]),
                )
                innovations = center + spread * stats.norm.ppf(probabilities)
            else:
                innovations = np.quantile(
                    fitted["residual_groups"].get(int(group), fallback), probabilities
                )
            ensemble[selected] = (
                q[selected, None] + mean_increment[selected, None]
                + sigma[selected, None] * innovations[None, :]
            )
    else:
        raise KeyError(model)
    return np.maximum(ensemble, 0.0)


def ensemble_crps(ensemble: np.ndarray, observation: np.ndarray) -> np.ndarray:
    values = np.sort(ensemble, axis=1)
    members = values.shape[1]
    first = np.mean(np.abs(values - observation[:, None]), axis=1)
    coefficient = 2 * np.arange(1, members + 1) - members - 1
    second = np.sum(values * coefficient[None, :], axis=1) / members**2
    return first - second


def probability_metrics(probability: np.ndarray, target: np.ndarray, clip: float) -> dict:
    p = np.clip(np.asarray(probability, dtype=float), clip, 1 - clip)
    y = np.asarray(target, dtype=bool)
    return {
        "n": len(y),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "brier": float(np.mean((p - y.astype(float)) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (~y) * np.log(1 - p))),
        "probability_bias": float(np.mean(p) - y.mean()),
        "mean_lowflow_probability": float(np.mean(p)),
    }


def continuous_metrics(ensemble: np.ndarray, observation: np.ndarray, nominal: float) -> dict:
    alpha = (1 - nominal) / 2
    low = np.quantile(ensemble, alpha, axis=1)
    high = np.quantile(ensemble, 1 - alpha, axis=1)
    median = np.median(ensemble, axis=1)
    coverage = float(np.mean((observation >= low) & (observation <= high)))
    return {
        "crps": float(np.mean(ensemble_crps(ensemble, observation))),
        "interval_coverage": coverage,
        "interval_coverage_abs_error": float(abs(coverage - nominal)),
        "interval_mean_width": float(np.mean(high - low)),
        "median_mae": float(np.mean(np.abs(median - observation))),
    }


def qbin_probabilities(
    train_q: np.ndarray, train_target: np.ndarray, test_q: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    train_group = np.searchsorted(edges[1:-1], train_q, side="right")
    test_group = np.searchsorted(edges[1:-1], test_q, side="right")
    global_probability = (train_target.sum() + 0.5) / (len(train_target) + 1.0)
    group_probability = {}
    for group in range(len(edges) - 1):
        selected = train_group == group
        group_probability[group] = (
            (train_target[selected].sum() + 0.5) / (selected.sum() + 1.0)
            if selected.any() else global_probability
        )
    return np.asarray([group_probability[int(group)] for group in test_group], dtype=float)


def probability_score_arrays(
    probability: np.ndarray, target: np.ndarray, clip: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return observation-level Brier and logarithmic scores."""
    p = np.clip(np.asarray(probability, dtype=float), clip, 1 - clip)
    y = np.asarray(target, dtype=bool)
    return (
        (p - y.astype(float)) ** 2,
        -(y * np.log(p) + (~y) * np.log(1 - p)),
    )


def water_year(dates: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(dates, errors="coerce")
    return (dates.dt.year + dates.dt.month.ge(10).astype(int)).to_numpy(int)


def basin_worker(task):
    (
        gage, source, spatial_group, output_file, exclusion_file,
        water_year_file, cfg, force,
    ) = task
    output = Path(output_file)
    exclusion_path = Path(exclusion_file)
    water_year_path = Path(water_year_file)
    if output.exists() and exclusion_path.exists() and water_year_path.exists() and not force:
        cached = pd.read_csv(output, dtype={"GAGE_ID": str})
        if (
            len(cached) and "analysis_version" in cached
            and int(cached["analysis_version"].iloc[0]) >= ANALYSIS_VERSION
        ):
            return (
                cached,
                pd.read_csv(exclusion_path, dtype={"GAGE_ID": str}),
                pd.read_csv(water_year_path, dtype={"GAGE_ID": str}),
            )

    setting = cfg["predictability_test"]
    frame = pd.read_csv(source, parse_dates=["date"], low_memory=False).sort_values("date")
    training_end = pd.Timestamp(setting["training_end_date"])
    evaluation_start = pd.Timestamp(setting["evaluation_start_date"])
    q_raw = pd.to_numeric(frame["q_mm_day"], errors="coerce")
    scale_source = q_raw[(frame["date"] <= training_end) & q_raw.gt(0)]
    scale = float(scale_source.median())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("no_positive_pre_evaluation_discharge_scale")
    frame["q_norm_predictability"] = q_raw / scale
    feature_frame, feature_columns = initialization_features(frame, scale, cfg)
    historical = frame.loc[frame["date"] <= training_end, "q_norm_predictability"]
    thresholds = {
        float(quantile): float(historical.quantile(float(quantile)))
        for quantile in setting["low_flow_quantiles"]
    }

    rows, exclusions, water_year_rows = [], [], []
    for lead_value in setting["lead_days"]:
        lead = int(lead_value)
        transitions = dry_transition_frame(frame, lead, cfg).merge(
            feature_frame, on="date", how="left", validate="many_to_one"
        )
        train = transitions[transitions["future_date"] <= training_end].copy()
        test = transitions[transitions["date"] >= evaluation_start].copy()
        if len(train) < int(setting["minimum_training_transitions"]):
            exclusions.append({
                "GAGE_ID": str(gage).zfill(8), "lead_days": lead,
                "threshold_name": "ALL", "reason": "insufficient_training_transitions",
            })
            continue
        if len(test) < int(setting["minimum_test_transitions"]):
            exclusions.append({
                "GAGE_ID": str(gage).zfill(8), "lead_days": lead,
                "threshold_name": "ALL", "reason": "insufficient_evaluation_transitions",
            })
            continue
        try:
            fitted = fit_transition_model(train, cfg)
        except ValueError as error:
            exclusions.append({
                "GAGE_ID": str(gage).zfill(8), "lead_days": lead,
                "threshold_name": "ALL", "reason": str(error),
            })
            continue

        q_test = test["q"].to_numpy(float)
        observed = test["q_next"].to_numpy(float)
        q_train = train["q"].to_numpy(float)
        train_observed = train["q_next"].to_numpy(float)
        ensembles = {
            model: distribution_members(model, q_test, fitted, int(setting["ensemble_members"]))
            for model in [GAUSSIAN_MODEL, STATE_GAUSSIAN_MODEL, MAIN_MODEL]
        }
        common = {
            "analysis_version": ANALYSIS_VERSION,
            "GAGE_ID": str(gage).zfill(8),
            "spatial_group": spatial_group,
            "lead_days": lead,
            "training_transitions": len(train),
            "test_transitions_available": len(test),
            "raw_variance_exponent": fitted["raw_variance_exponent"],
            "variance_exponent_standard_error": fitted["variance_exponent_standard_error"],
            "regularized_variance_exponent": fitted["regularized_variance_exponent"],
            "variance_coefficient": fitted["variance_coefficient"],
            "variance_r2": fitted["variance_r2"],
            "q_model_low": fitted["q_model_low"],
            "q_model_high": fitted["q_model_high"],
        }
        for quantile, threshold in thresholds.items():
            name = threshold_name(quantile)
            if not np.isfinite(threshold) or (
                setting["require_positive_low_flow_threshold"] and threshold <= 0
            ):
                exclusions.append({
                    "GAGE_ID": str(gage).zfill(8), "lead_days": lead,
                    "threshold_name": name, "reason": "nonpositive_training_threshold",
                })
                continue
            train_target = train_observed <= threshold
            test_target = observed <= threshold
            events = int(test_target.sum())
            nonevents = int((~test_target).sum())
            if events < int(setting["minimum_events_per_basin"]):
                exclusions.append({
                    "GAGE_ID": str(gage).zfill(8), "lead_days": lead,
                    "threshold_name": name, "reason": "insufficient_evaluation_events",
                })
                continue
            if nonevents < int(setting["minimum_nonevents_per_basin"]):
                exclusions.append({
                    "GAGE_ID": str(gage).zfill(8), "lead_days": lead,
                    "threshold_name": name, "reason": "insufficient_evaluation_nonevents",
                })
                continue
            if train_target.sum() == 0 or (~train_target).sum() == 0:
                exclusions.append({
                    "GAGE_ID": str(gage).zfill(8), "lead_days": lead,
                    "threshold_name": name, "reason": "training_outcome_has_one_class",
                })
                continue

            clip = float(setting["probability_clip"])
            baseline_probabilities = {
                "training_climatology": np.full(
                    len(test_target), (train_target.sum() + 0.5) / (len(train_target) + 1.0)
                ),
                "persistence": (q_test <= threshold).astype(float),
                "qbin_probability": qbin_probabilities(
                    q_train, train_target, q_test, fitted["bin_edges"]
                ),
            }
            ensemble_probabilities = {
                model: (np.sum(ensemble <= threshold, axis=1) + 0.5)
                / (ensemble.shape[1] + 1.0)
                for model, ensemble in ensembles.items()
            }
            hybrid_weight = (
                float(setting["recession_logistic_blend_weight"])
                if lead >= int(setting["recession_logistic_activation_lead_days"])
                else 0.0
            )
            empirical_probability = ensemble_probabilities[MAIN_MODEL]
            if hybrid_weight > 0:
                logistic_probability = recession_logistic_probabilities(
                    train, test, train_target, feature_columns, cfg
                )
                operational_probability = (
                    (1.0 - hybrid_weight) * empirical_probability
                    + hybrid_weight * logistic_probability
                )
            else:
                operational_probability = empirical_probability.copy()
            row_common = {
                **common,
                "threshold_quantile": quantile,
                "threshold_name": name,
                "low_flow_threshold_qnorm": threshold,
                "training_events": int(train_target.sum()),
                "training_event_rate": float(train_target.mean()),
                "recession_logistic_blend_weight": hybrid_weight,
            }
            for model, probability in baseline_probabilities.items():
                rows.append({
                    **row_common, "model": model,
                    **probability_metrics(probability, test_target, clip),
                    "crps": np.nan, "interval_coverage": np.nan,
                    "interval_coverage_abs_error": np.nan,
                    "interval_mean_width": np.nan, "median_mae": np.nan,
                })
            for model, ensemble in ensembles.items():
                probability = ensemble_probabilities[model]
                rows.append({
                    **row_common, "model": model,
                    **probability_metrics(probability, test_target, clip),
                    **continuous_metrics(
                        ensemble, observed, float(setting["nominal_interval_coverage"])
                    ),
                })
            rows.append({
                **row_common, "model": OPERATIONAL_MODEL,
                **probability_metrics(operational_probability, test_target, clip),
                "crps": np.nan, "interval_coverage": np.nan,
                "interval_coverage_abs_error": np.nan,
                "interval_mean_width": np.nan, "median_mae": np.nan,
            })

            all_probabilities = {
                **baseline_probabilities, **ensemble_probabilities,
                OPERATIONAL_MODEL: operational_probability,
            }
            scores = {
                model: probability_score_arrays(probability, test_target, clip)
                for model, probability in all_probabilities.items()
            }
            crps = {
                model: ensemble_crps(ensemble, observed)
                for model, ensemble in ensembles.items()
            }
            years = water_year(test["future_date"])
            for year in sorted(np.unique(years)):
                selected_year = years == year
                if not selected_year.any():
                    continue
                main_brier, main_log = scores[MAIN_MODEL]
                operational_brier, operational_log = scores[OPERATIONAL_MODEL]
                global_brier, global_log = scores[GAUSSIAN_MODEL]
                state_brier, state_log = scores[STATE_GAUSSIAN_MODEL]
                water_year_rows.append({
                    "analysis_version": ANALYSIS_VERSION,
                    "GAGE_ID": str(gage).zfill(8),
                    "spatial_group": spatial_group,
                    "threshold_quantile": quantile,
                    "threshold_name": name,
                    "lead_days": lead,
                    "water_year": int(year),
                    "n": int(selected_year.sum()),
                    "events": int(test_target[selected_year].sum()),
                    **{
                        f"brier_improvement_vs_{reference}": float(np.mean(
                            scores[reference][0][selected_year]
                            - operational_brier[selected_year]
                        ))
                        for reference in BASELINE_MODELS
                    },
                    **{
                        f"brier_score_{reference}": float(np.mean(
                            scores[reference][0][selected_year]
                        ))
                        for reference in BASELINE_MODELS
                    },
                    "brier_score_operational": float(np.mean(
                        operational_brier[selected_year]
                    )),
                    **{
                        f"log_loss_improvement_vs_{reference}": float(np.mean(
                            scores[reference][1][selected_year]
                            - operational_log[selected_year]
                        ))
                        for reference in BASELINE_MODELS
                    },
                    "brier_improvement_operational_vs_empirical": float(np.mean(
                        main_brier[selected_year] - operational_brier[selected_year]
                    )),
                    "log_loss_improvement_operational_vs_empirical": float(np.mean(
                        main_log[selected_year] - operational_log[selected_year]
                    )),
                    "brier_improvement_state_gaussian_vs_global_gaussian": float(np.mean(
                        global_brier[selected_year] - state_brier[selected_year]
                    )),
                    "log_loss_improvement_state_gaussian_vs_global_gaussian": float(np.mean(
                        global_log[selected_year] - state_log[selected_year]
                    )),
                    "crps_improvement_state_gaussian_vs_global_gaussian": float(np.mean(
                        crps[GAUSSIAN_MODEL][selected_year]
                        - crps[STATE_GAUSSIAN_MODEL][selected_year]
                    )),
                    "brier_improvement_empirical_vs_state_gaussian": float(np.mean(
                        state_brier[selected_year] - main_brier[selected_year]
                    )),
                    "log_loss_improvement_empirical_vs_state_gaussian": float(np.mean(
                        state_log[selected_year] - main_log[selected_year]
                    )),
                    "crps_improvement_empirical_vs_state_gaussian": float(np.mean(
                        crps[STATE_GAUSSIAN_MODEL][selected_year]
                        - crps[MAIN_MODEL][selected_year]
                    )),
                })

    result = pd.DataFrame(rows)
    exclusion_frame = pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS)
    water_year_frame = pd.DataFrame(water_year_rows)
    if water_year_frame.empty:
        water_year_frame = pd.DataFrame(columns=["analysis_version", "GAGE_ID"])
    output.parent.mkdir(parents=True, exist_ok=True)
    exclusion_path.parent.mkdir(parents=True, exist_ok=True)
    water_year_path.parent.mkdir(parents=True, exist_ok=True)
    if not result.empty:
        with atomic_target(output) as temporary:
            result.to_csv(temporary, index=False)
    with atomic_target(exclusion_path) as temporary:
        exclusion_frame.to_csv(temporary, index=False)
    with atomic_target(water_year_path) as temporary:
        water_year_frame.to_csv(temporary, index=False)
    return result, exclusion_frame, water_year_frame


def hierarchical_bootstrap(
    frame: pd.DataFrame, replicates: int, chunk_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Resample spatial groups, then basins within sampled groups."""
    groups = sorted(frame["spatial_group"].astype(str).unique())
    output = np.empty(replicates, dtype=float)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        size = stop - start
        numerators = np.empty((len(groups), size), dtype=float)
        denominators = np.empty((len(groups), size), dtype=float)
        for group_index, group in enumerate(groups):
            selected = frame[
                frame["spatial_group"].astype(str).eq(group)
            ].sort_values("GAGE_ID")
            values = selected["improvement"].to_numpy(float)
            weights = selected["weight"].to_numpy(float)
            indices = rng.integers(0, len(selected), size=(size, len(selected)))
            numerators[group_index] = np.sum(values[indices] * weights[indices], axis=1)
            denominators[group_index] = np.sum(weights[indices], axis=1)
        sampled_groups = rng.integers(0, len(groups), size=(size, len(groups)))
        replicate_index = np.arange(size)[:, None]
        numerator = numerators[sampled_groups, replicate_index].sum(axis=1)
        denominator = denominators[sampled_groups, replicate_index].sum(axis=1)
        output[start:stop] = numerator / denominator
    return output


def paired_skill(metrics: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    comparisons = [
        (OPERATIONAL_MODEL, reference, ["brier", "log_loss"])
        for reference in BASELINE_MODELS
    ] + [
        (OPERATIONAL_MODEL, MAIN_MODEL, ["brier", "log_loss"]),
        (MAIN_MODEL, GAUSSIAN_MODEL, [
            "brier", "log_loss", "crps", "interval_coverage_abs_error", "median_mae"
        ]),
        (STATE_GAUSSIAN_MODEL, GAUSSIAN_MODEL, [
            "brier", "log_loss", "crps", "interval_coverage_abs_error", "median_mae"
        ]),
        (MAIN_MODEL, STATE_GAUSSIAN_MODEL, [
            "brier", "log_loss", "crps", "interval_coverage_abs_error", "median_mae"
        ]),
    ]
    directions = {
        "brier": -1.0, "log_loss": -1.0, "crps": -1.0,
        "interval_coverage_abs_error": -1.0, "median_mae": -1.0,
    }
    setting = cfg["predictability_test"]
    rng = np.random.default_rng(cfg["project"]["random_seed"] + 18_000_000)
    replicates = int(setting["paired_bootstrap_replicates"])
    chunk_size = int(setting["bootstrap_chunk_size"])
    keys = [
        "GAGE_ID", "spatial_group", "threshold_quantile", "threshold_name", "lead_days"
    ]
    rows = []
    for model, reference, metric_names in comparisons:
        alternative = metrics[metrics["model"].eq(model)]
        baseline = metrics[metrics["model"].eq(reference)]
        paired = baseline.merge(alternative, on=keys, suffixes=("_reference", "_model"))
        for (quantile, name, lead), lead_data in paired.groupby([
            "threshold_quantile", "threshold_name", "lead_days"
        ]):
            for metric in metric_names:
                finite = lead_data[
                    np.isfinite(lead_data[f"{metric}_reference"])
                    & np.isfinite(lead_data[f"{metric}_model"])
                ].copy()
                if finite.empty:
                    continue
                finite["improvement"] = directions[metric] * (
                    finite[f"{metric}_model"] - finite[f"{metric}_reference"]
                )
                finite["weight"] = finite["n_reference"].to_numpy(float)
                estimate = float(np.average(finite["improvement"], weights=finite["weight"]))
                regional = finite.groupby("spatial_group")["improvement"].mean()
                boot = hierarchical_bootstrap(finite, replicates, chunk_size, rng)
                low, high = np.quantile(boot, [0.025, 0.975])
                reference_score = float(np.average(
                    finite[f"{metric}_reference"], weights=finite["weight"]
                ))
                rows.append({
                    "threshold_quantile": float(quantile),
                    "threshold_name": name,
                    "lead_days": int(lead),
                    "model": model,
                    "reference_model": reference,
                    "metric": metric,
                    "basins": len(finite),
                    "spatial_groups": finite["spatial_group"].nunique(),
                    "test_transitions": int(finite["weight"].sum()),
                    "weighted_improvement": estimate,
                    "relative_skill": (
                        estimate / reference_score if reference_score > 0 else np.nan
                    ),
                    "equal_group_improvement": float(regional.mean()),
                    "bootstrap_ci_low": float(low),
                    "bootstrap_ci_high": float(high),
                    "basins_improved": int((finite["improvement"] > 0).sum()),
                    "groups_improved": int((regional > 0).sum()),
                    "positive_means_better": True,
                })
    return pd.DataFrame(rows)


def water_year_hierarchical_bootstrap(
    frame: pd.DataFrame, replicates: int, chunk_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, pd.DataFrame]:
    """Resample water-year blocks, spatial groups, and basins within groups."""
    minimum_years = int(frame.attrs.get("minimum_water_years", 3))
    counts = frame.groupby("GAGE_ID")["water_year"].nunique()
    eligible = counts[counts >= minimum_years].index
    work = frame[frame["GAGE_ID"].isin(eligible)].copy()
    if work.empty:
        return np.array([], dtype=float), work
    work["spatial_group"] = work["spatial_group"].astype(str)
    work["weighted_improvement"] = work["improvement"] * work["weight"]
    work = work.groupby(
        ["GAGE_ID", "spatial_group", "water_year"], as_index=False
    ).agg(weighted_improvement=("weighted_improvement", "sum"), weight=("weight", "sum"))
    work["improvement"] = work["weighted_improvement"] / work["weight"]
    basins = work[["GAGE_ID", "spatial_group"]].drop_duplicates().sort_values("GAGE_ID")
    years = np.sort(work["water_year"].unique())
    basin_index = pd.Index(basins["GAGE_ID"])
    value_matrix = work.pivot(index="GAGE_ID", columns="water_year", values="improvement")
    weight_matrix = work.pivot(index="GAGE_ID", columns="water_year", values="weight")
    values = value_matrix.reindex(index=basin_index, columns=years).to_numpy(float)
    weights = weight_matrix.reindex(index=basin_index, columns=years).to_numpy(float)
    values = np.nan_to_num(values, nan=0.0)
    weights = np.nan_to_num(weights, nan=0.0)
    group_labels = basins.set_index("GAGE_ID").loc[basin_index, "spatial_group"].to_numpy(str)
    groups = np.unique(group_labels)
    group_members = [np.flatnonzero(group_labels == group) for group in groups]
    output = np.empty(replicates, dtype=float)
    chunk_size = min(int(chunk_size), 64)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        size = stop - start
        sampled_years = rng.integers(0, len(years), size=(size, len(years)))
        selected_values = values[:, sampled_years]
        selected_weights = weights[:, sampled_years]
        basin_numerator = np.sum(selected_values * selected_weights, axis=2).T
        basin_denominator = np.sum(selected_weights, axis=2).T
        group_numerator = np.zeros((size, len(groups)), dtype=float)
        group_denominator = np.zeros((size, len(groups)), dtype=float)
        for group_index, members in enumerate(group_members):
            sampled_basins = rng.integers(0, len(members), size=(size, len(members)))
            actual = members[sampled_basins]
            row_index = np.arange(size)[:, None]
            group_numerator[:, group_index] = basin_numerator[row_index, actual].sum(axis=1)
            group_denominator[:, group_index] = basin_denominator[row_index, actual].sum(axis=1)
        sampled_groups = rng.integers(0, len(groups), size=(size, len(groups)))
        row_index = np.arange(size)[:, None]
        numerator = group_numerator[row_index, sampled_groups].sum(axis=1)
        denominator = group_denominator[row_index, sampled_groups].sum(axis=1)
        output[start:stop] = np.divide(
            numerator, denominator,
            out=np.full(size, np.nan, dtype=float), where=denominator > 0,
        )
    return output[np.isfinite(output)], work


def water_year_block_skill(yearly: pd.DataFrame, cfg: dict, logger=None) -> pd.DataFrame:
    """Test key model contrasts while preserving annual temporal dependence."""
    comparisons = [
        (OPERATIONAL_MODEL, reference, "brier", f"brier_improvement_vs_{reference}")
        for reference in BASELINE_MODELS
    ] + [
        (
            OPERATIONAL_MODEL, MAIN_MODEL, "brier",
            "brier_improvement_operational_vs_empirical",
        ),
        (
            STATE_GAUSSIAN_MODEL, GAUSSIAN_MODEL, "crps",
            "crps_improvement_state_gaussian_vs_global_gaussian",
        ),
        (
            MAIN_MODEL, STATE_GAUSSIAN_MODEL, "crps",
            "crps_improvement_empirical_vs_state_gaussian",
        ),
    ]
    setting = cfg["predictability_test"]
    rng = np.random.default_rng(cfg["project"]["random_seed"] + 18_500_000)
    replicates = int(setting["water_year_block_bootstrap_replicates"])
    chunk_size = int(setting["bootstrap_chunk_size"])
    minimum_years = int(setting["minimum_evaluation_water_years_per_basin"])
    rows = []
    grouped = list(yearly.groupby([
        "threshold_quantile", "threshold_name", "lead_days"
    ]))
    for group_number, ((quantile, name, lead), lead_data) in enumerate(grouped, start=1):
        for model, reference, metric, column in comparisons:
            if column not in lead_data:
                continue
            selected = lead_data[
                np.isfinite(lead_data[column]) & lead_data["n"].gt(0)
            ].copy()
            selected["improvement"] = selected[column].to_numpy(float)
            selected["weight"] = selected["n"].to_numpy(float)
            selected.attrs["minimum_water_years"] = minimum_years
            boot, eligible = water_year_hierarchical_bootstrap(
                selected, replicates, chunk_size, rng
            )
            if eligible.empty or len(boot) < max(100, replicates // 2):
                continue
            low, high = np.quantile(boot, [0.025, 0.975])
            estimate = float(np.average(eligible["improvement"], weights=eligible["weight"]))
            basin_scores = eligible.assign(
                weighted=eligible["improvement"] * eligible["weight"]
            ).groupby("GAGE_ID").agg(weighted=("weighted", "sum"), weight=("weight", "sum"))
            basin_means = basin_scores["weighted"] / basin_scores["weight"]
            rows.append({
                "threshold_quantile": float(quantile), "threshold_name": name,
                "lead_days": int(lead), "model": model,
                "reference_model": reference, "metric": metric,
                "basins": int(eligible["GAGE_ID"].nunique()),
                "spatial_groups": int(eligible["spatial_group"].nunique()),
                "water_years": int(eligible["water_year"].nunique()),
                "test_transitions": int(eligible["weight"].sum()),
                "weighted_improvement": estimate,
                "bootstrap_ci_low": float(low), "bootstrap_ci_high": float(high),
                "basins_improved": int((basin_means > 0).sum()),
                "positive_means_better": True,
                "resampling_unit": "water_year_then_spatial_group_then_basin",
            })
        if logger is not None:
            logger.info(
                "Water-year block bootstrap %d/%d: %s, lead %d days",
                group_number, len(grouped), name, int(lead),
            )
    return pd.DataFrame(rows)


def severity_skill_bootstrap(
    frame: pd.DataFrame, replicates: int, chunk_size: int,
    minimum_years: int, rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Bootstrap a difference between ratios of aggregated Brier scores."""
    counts = frame.groupby("GAGE_ID")["water_year"].nunique()
    eligible_gages = counts[counts >= minimum_years].index
    work = frame[frame["GAGE_ID"].isin(eligible_gages)].copy()
    if work.empty:
        return np.array([], dtype=float), work
    for threshold in ["extreme", "comparison"]:
        n = work[f"n_{threshold}"].to_numpy(float)
        work[f"model_sum_{threshold}"] = (
            work[f"brier_score_operational_{threshold}"].to_numpy(float) * n
        )
        work[f"reference_sum_{threshold}"] = (
            work[f"brier_score_reference_{threshold}"].to_numpy(float) * n
        )
    sum_columns = [
        "model_sum_extreme", "reference_sum_extreme",
        "model_sum_comparison", "reference_sum_comparison",
    ]
    work = work.groupby(
        ["GAGE_ID", "spatial_group", "water_year"], as_index=False
    )[sum_columns].sum()
    basins = work[["GAGE_ID", "spatial_group"]].drop_duplicates().sort_values("GAGE_ID")
    years = np.sort(work["water_year"].unique())
    basin_index = pd.Index(basins["GAGE_ID"])
    matrices = {
        column: np.nan_to_num(
            work.pivot(index="GAGE_ID", columns="water_year", values=column)
            .reindex(index=basin_index, columns=years).to_numpy(float), nan=0.0,
        )
        for column in sum_columns
    }
    group_labels = basins.set_index("GAGE_ID").loc[
        basin_index, "spatial_group"
    ].astype(str).to_numpy()
    groups = np.unique(group_labels)
    group_members = [np.flatnonzero(group_labels == group) for group in groups]
    output = np.full(replicates, np.nan, dtype=float)
    chunk_size = min(int(chunk_size), 64)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        size = stop - start
        sampled_years = rng.integers(0, len(years), size=(size, len(years)))
        basin_totals = {
            column: np.sum(matrix[:, sampled_years], axis=2).T
            for column, matrix in matrices.items()
        }
        group_totals = {
            column: np.zeros((size, len(groups)), dtype=float)
            for column in sum_columns
        }
        row_index = np.arange(size)[:, None]
        for group_index, members in enumerate(group_members):
            sampled_basins = rng.integers(0, len(members), size=(size, len(members)))
            actual = members[sampled_basins]
            for column in sum_columns:
                group_totals[column][:, group_index] = basin_totals[column][
                    row_index, actual
                ].sum(axis=1)
        sampled_groups = rng.integers(0, len(groups), size=(size, len(groups)))
        totals = {
            column: values[row_index, sampled_groups].sum(axis=1)
            for column, values in group_totals.items()
        }
        extreme_ratio = np.divide(
            totals["model_sum_extreme"], totals["reference_sum_extreme"],
            out=np.full(size, np.nan), where=totals["reference_sum_extreme"] > 0,
        )
        comparison_ratio = np.divide(
            totals["model_sum_comparison"], totals["reference_sum_comparison"],
            out=np.full(size, np.nan), where=totals["reference_sum_comparison"] > 0,
        )
        output[start:stop] = comparison_ratio - extreme_ratio
    return output[np.isfinite(output)], work


def severity_contrasts(yearly: pd.DataFrame, cfg: dict, logger=None) -> pd.DataFrame:
    """Compare Q5 and Q10/Q20 using stable aggregate Brier skill scores."""
    reference = str(cfg["predictability_test"]["severity_contrast_reference_model"])
    operational_column = "brier_score_operational"
    reference_column = f"brier_score_{reference}"
    extreme_quantile = min(float(value) for value in cfg["predictability_test"]["low_flow_quantiles"])
    extreme = yearly[yearly["threshold_quantile"].eq(extreme_quantile)]
    keys = ["GAGE_ID", "spatial_group", "lead_days", "water_year"]
    setting = cfg["predictability_test"]
    rng = np.random.default_rng(cfg["project"]["random_seed"] + 18_750_000)
    rows = []
    for comparison_quantile in sorted(
        value for value in yearly["threshold_quantile"].unique() if value > extreme_quantile
    ):
        comparison = yearly[yearly["threshold_quantile"].eq(comparison_quantile)]
        paired = extreme[keys + ["n", operational_column, reference_column]].rename(
            columns={reference_column: "brier_score_reference"}
        ).merge(
            comparison[keys + ["n", operational_column, reference_column]].rename(
                columns={reference_column: "brier_score_reference"}
            ),
            on=keys, suffixes=("_extreme", "_comparison")
        )
        for lead, lead_data in paired.groupby("lead_days"):
            required = [
                "brier_score_operational_extreme", "brier_score_reference_extreme",
                "brier_score_operational_comparison", "brier_score_reference_comparison",
            ]
            finite = np.isfinite(lead_data[required]).all(axis=1)
            lead_data = lead_data[
                finite & lead_data["n_extreme"].gt(0)
                & lead_data["n_comparison"].gt(0)
            ].copy()
            boot, eligible = severity_skill_bootstrap(
                lead_data,
                int(setting["water_year_block_bootstrap_replicates"]),
                int(setting["bootstrap_chunk_size"]),
                int(setting["minimum_evaluation_water_years_per_basin"]), rng,
            )
            if eligible.empty or len(boot) < 100:
                continue
            low, high = np.quantile(boot, [0.025, 0.975])
            extreme_skill = 1.0 - (
                eligible["model_sum_extreme"].sum()
                / eligible["reference_sum_extreme"].sum()
            )
            comparison_skill = 1.0 - (
                eligible["model_sum_comparison"].sum()
                / eligible["reference_sum_comparison"].sum()
            )
            rows.append({
                "extreme_threshold_quantile": extreme_quantile,
                "extreme_threshold_name": threshold_name(extreme_quantile),
                "comparison_threshold_quantile": float(comparison_quantile),
                "comparison_threshold_name": threshold_name(float(comparison_quantile)),
                "reference_model": reference, "lead_days": int(lead),
                "basins": int(eligible["GAGE_ID"].nunique()),
                "spatial_groups": int(eligible["spatial_group"].nunique()),
                "water_years": int(eligible["water_year"].nunique()),
                "weighted_q5_minus_comparison_brier_skill": float(
                    extreme_skill - comparison_skill
                ),
                "bootstrap_ci_low": float(low), "bootstrap_ci_high": float(high),
                "positive_means_q5_retains_more_skill": True,
            })
            if logger is not None:
                logger.info(
                    "Severity block bootstrap: Q5 versus %s, lead %d days",
                    threshold_name(float(comparison_quantile)), int(lead),
                )
    return pd.DataFrame(rows, columns=SEVERITY_COLUMNS)


def model_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "event_rate", "brier", "log_loss", "probability_bias", "crps",
        "interval_coverage", "interval_coverage_abs_error", "interval_mean_width",
        "median_mae", "mean_lowflow_probability",
    ]
    rows = []
    for (quantile, name, lead, model), group in metrics.groupby([
        "threshold_quantile", "threshold_name", "lead_days", "model"
    ]):
        weights = group["n"].to_numpy(float)
        row = {
            "threshold_quantile": float(quantile), "threshold_name": name,
            "lead_days": int(lead), "model": model, "basins": len(group),
            "spatial_groups": group["spatial_group"].nunique(),
            "test_transitions": int(weights.sum()), "events": int(group["events"].sum()),
        }
        for column in numeric:
            finite = np.isfinite(group[column])
            row[column] = (
                float(np.average(group.loc[finite, column], weights=weights[finite]))
                if finite.any() else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def basin_horizons(metrics: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    setting = cfg["predictability_test"]
    leads = [int(value) for value in setting["lead_days"]]
    index = ["GAGE_ID", "spatial_group", "threshold_quantile", "threshold_name", "lead_days"]
    wide = metrics.pivot_table(index=index, columns="model", values="brier", aggfunc="first")
    wide = wide.reset_index()
    rows = []
    for (gage, group, quantile, name), basin in wide.groupby(index[:-1]):
        basin = basin.set_index("lead_days")
        references = [*BASELINE_MODELS, "all_baselines"]
        for reference in references:
            improvements = {}
            for lead in leads:
                if lead not in basin.index or OPERATIONAL_MODEL not in basin.columns:
                    continue
                current = basin.loc[lead]
                if reference == "all_baselines":
                    if not all(item in basin.columns and np.isfinite(current[item]) for item in BASELINE_MODELS):
                        continue
                    improvements[lead] = min(
                        float(current[item] - current[OPERATIONAL_MODEL])
                        for item in BASELINE_MODELS
                    )
                elif reference in basin.columns and np.isfinite(current[reference]):
                    improvements[lead] = float(
                        current[reference] - current[OPERATIONAL_MODEL]
                    )
            horizon = 0
            first_failure = None
            for lead in leads:
                if lead not in improvements or improvements[lead] <= 0:
                    first_failure = lead
                    break
                horizon = lead
            evaluable = sorted(improvements)
            rows.append({
                "GAGE_ID": str(gage).zfill(8), "spatial_group": group,
                "threshold_quantile": float(quantile), "threshold_name": name,
                "reference_model": reference,
                "predictability_horizon_days": int(horizon),
                "maximum_evaluable_lead_days": int(max(evaluable)) if evaluable else 0,
                "first_nonpositive_or_missing_lead_days": first_failure,
                "right_censored_at_30_days": bool(horizon == leads[-1]),
                "positive_skill_at_first_lead": bool(improvements.get(leads[0], -np.inf) > 0),
            })
    return pd.DataFrame(rows)


def continental_horizons(paired: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    setting = cfg["predictability_test"]
    leads = [int(value) for value in setting["lead_days"]]
    minimum_basins = int(setting["minimum_basins_per_lead_for_claim"])
    minimum_groups = int(setting["minimum_spatial_groups_per_lead_for_claim"])
    brier = paired[
        paired["model"].eq(OPERATIONAL_MODEL) & paired["metric"].eq("brier")
        & paired["reference_model"].isin(BASELINE_MODELS)
    ].copy()
    decisions = []
    for quantile in sorted(brier["threshold_quantile"].unique(), reverse=True):
        name = threshold_name(float(quantile))
        for reference in [*BASELINE_MODELS, "all_baselines"]:
            for lead in leads:
                if reference == "all_baselines":
                    selected = brier[
                        brier["threshold_quantile"].eq(quantile)
                        & brier["lead_days"].eq(lead)
                    ]
                    complete = set(selected["reference_model"]) == set(BASELINE_MODELS)
                    if complete:
                        estimate = float(selected["weighted_improvement"].min())
                        ci_low = float(selected["bootstrap_ci_low"].min())
                        ci_high = float(selected["bootstrap_ci_high"].min())
                        basins = int(selected["basins"].min())
                        groups = int(selected["spatial_groups"].min())
                    else:
                        estimate = ci_low = ci_high = np.nan
                        basins = groups = 0
                else:
                    selected = brier[
                        brier["threshold_quantile"].eq(quantile)
                        & brier["lead_days"].eq(lead)
                        & brier["reference_model"].eq(reference)
                    ]
                    if len(selected):
                        item = selected.iloc[0]
                        estimate = float(item["weighted_improvement"])
                        ci_low = float(item["bootstrap_ci_low"])
                        ci_high = float(item["bootstrap_ci_high"])
                        basins = int(item["basins"])
                        groups = int(item["spatial_groups"])
                    else:
                        estimate = ci_low = ci_high = np.nan
                        basins = groups = 0
                eligible = bool(basins >= minimum_basins and groups >= minimum_groups)
                positive = bool(eligible and np.isfinite(ci_low) and ci_low > 0)
                decisions.append({
                    "threshold_quantile": float(quantile), "threshold_name": name,
                    "reference_model": reference, "lead_days": lead,
                    "weighted_brier_improvement": estimate,
                    "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
                    "basins": basins, "spatial_groups": groups,
                    "sample_requirement_met": eligible,
                    "positive_skill_ci_excludes_zero": positive,
                })
    lead_table = pd.DataFrame(decisions)
    horizons = []
    for (quantile, name, reference), group in lead_table.groupby([
        "threshold_quantile", "threshold_name", "reference_model"
    ]):
        group = group.set_index("lead_days")
        horizon = 0
        first_failure = None
        for lead in leads:
            if lead not in group.index or not bool(
                group.loc[lead, "positive_skill_ci_excludes_zero"]
            ):
                first_failure = lead
                break
            horizon = lead
        evaluable = group[group["sample_requirement_met"]].index.to_list()
        horizons.append({
            "threshold_quantile": float(quantile), "threshold_name": name,
            "reference_model": reference,
            "predictability_horizon_days": int(horizon),
            "maximum_evaluable_lead_days": int(max(evaluable)) if evaluable else 0,
            "first_nonpositive_or_ineligible_lead_days": first_failure,
            "right_censored_at_30_days": bool(horizon == leads[-1]),
            "definition": (
                "largest uninterrupted configured lead with positive basin-paired Brier "
                "improvement whose hierarchical 95% CI excludes zero"
            ),
        })
    return lead_table, pd.DataFrame(horizons)


def summarize_basin_horizons(horizons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (quantile, name, reference), group in horizons.groupby([
        "threshold_quantile", "threshold_name", "reference_model"
    ]):
        values = group["predictability_horizon_days"].to_numpy(float)
        rows.append({
            "threshold_quantile": float(quantile), "threshold_name": name,
            "reference_model": reference, "basins": len(group),
            "median_horizon_days": float(np.median(values)),
            "p25_horizon_days": float(np.quantile(values, 0.25)),
            "p75_horizon_days": float(np.quantile(values, 0.75)),
            "fraction_positive_at_first_lead": float(group["positive_skill_at_first_lead"].mean()),
            "fraction_right_censored_at_30_days": float(group["right_censored_at_30_days"].mean()),
            "median_maximum_evaluable_lead_days": float(
                group["maximum_evaluable_lead_days"].median()
            ),
        })
    return pd.DataFrame(rows)


def write_analysis_summary(path: Path, summary: dict) -> None:
    lines = [
        "# Stage 18 analysis summary", "",
        "This is a temporally held-out, conditional no-effective-input scenario test. "
        "Future precipitation and proxy snowmelt are observed only to identify evaluation periods; "
        "this is not an unconditional operational forecast.", "",
        f"**Core dry-state dynamics criteria met:** {summary['core_dynamics_supported']}",
        f"**Combined skill criteria met:** {summary['combined_skill_criteria_supported']}",
        f"**Robust Q10 all-baseline predictability horizon:** {summary['primary_all_baseline_horizon_days']} days",
        f"**At least 7 days of incremental Q10 predictability supported:** {summary['predictability_horizon_supported']}",
        f"**Flow-state dependence adds skill beyond a global Gaussian:** {summary['state_dependence_incremental_value_supported']}",
        f"**Empirical tail shape adds skill beyond a state-specific Gaussian:** {summary['non_gaussian_incremental_value_supported']}",
        f"**Lead-adaptive recession-memory hybrid adds skill:** {summary['lead_adaptive_hybrid_incremental_value_supported']}",
        f"**Q5-versus-Q10/Q20 severity gradient supported:** {summary['threshold_severity_gradient_supported']}",
        "",
        "A robust horizon is the shorter of the spatial hierarchy and water-year-block results. "
        "It stops at the first configured lead that is ineligible or whose 95% Brier-skill interval "
        "does not exclude zero; later isolated positive leads do not extend it.",
        "A negative or short horizon must be reported and must not be converted into a positive forecast claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 17)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    setting = cfg["predictability_test"]

    inventory = pd.read_csv(
        stage_dir(cfg, 10) / "transition_inventory.csv", dtype={"GAGE_ID": str}
    )
    attributes = pd.read_csv(
        stage_dir(cfg, 15) / "continental_basin_results.csv",
        dtype={"GAGE_ID": str}, low_memory=False,
    )
    distribution = pd.read_csv(
        stage_dir(cfg, 17) / "distribution_basin_diagnostics.csv",
        dtype={"GAGE_ID": str}, low_memory=False,
    )
    inventory["GAGE_ID"] = inventory["GAGE_ID"].str.zfill(8)
    attributes["GAGE_ID"] = attributes["GAGE_ID"].str.zfill(8)
    distribution["GAGE_ID"] = distribution["GAGE_ID"].str.zfill(8)
    attribute_index = attributes.set_index("GAGE_ID", drop=False)
    inventory = inventory[
        inventory["GAGE_ID"].isin(attribute_index.index)
    ].sort_values("GAGE_ID")
    if args.limit:
        inventory = inventory.head(args.limit)

    group_column = setting["spatial_group_column"]
    by_basin = out / "by_basin"
    by_basin_exclusions = out / "by_basin_exclusions"
    by_basin_water_year = out / "by_basin_water_year"
    tasks = []
    for row in inventory.itertuples(index=False):
        attribute = attribute_index.loc[row.GAGE_ID]
        group = attribute.get(group_column, attribute.get("ecoregion", "unknown"))
        group = "unknown" if pd.isna(group) or not str(group).strip() else str(group)
        tasks.append((
            row.GAGE_ID, row.file, group,
            str(by_basin / f"{row.GAGE_ID}.csv"),
            str(by_basin_exclusions / f"{row.GAGE_ID}.csv"),
            str(by_basin_water_year / f"{row.GAGE_ID}.csv"),
            cfg, args.force,
        ))

    frames, exclusion_frames, water_year_frames, failures = [], [], [], []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(basin_worker, task): task[0] for task in tasks}
        for number, future in enumerate(as_completed(futures), start=1):
            gage = futures[future]
            try:
                metrics, exclusions, yearly = future.result()
                if not metrics.empty:
                    frames.append(metrics)
                if not exclusions.empty:
                    exclusion_frames.append(exclusions)
                if not yearly.empty:
                    water_year_frames.append(yearly)
            except ValueError as error:
                exclusion_frames.append(pd.DataFrame([{
                    "GAGE_ID": gage, "lead_days": np.nan,
                    "threshold_name": "ALL", "reason": str(error),
                }]))
            except Exception as error:
                failures.append({"GAGE_ID": gage, "error": repr(error)})
                logger.error("Predictability-horizon test failed for %s: %r", gage, error)
            if number % 100 == 0:
                logger.info("Completed %d/%d basin predictability tests", number, len(tasks))

    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    exclusions = (
        pd.concat(exclusion_frames, ignore_index=True)
        if exclusion_frames else pd.DataFrame(columns=EXCLUSION_COLUMNS)
    )
    if metrics.empty:
        raise RuntimeError("No basin supported the Stage-18 predictability-horizon test")
    yearly = pd.concat(water_year_frames, ignore_index=True) if water_year_frames else pd.DataFrame()
    if yearly.empty:
        raise RuntimeError("No basin produced Stage-18 water-year score differences")
    metrics["GAGE_ID"] = metrics["GAGE_ID"].astype(str).str.zfill(8)
    yearly["GAGE_ID"] = yearly["GAGE_ID"].astype(str).str.zfill(8)
    metrics = metrics.sort_values([
        "GAGE_ID", "threshold_quantile", "lead_days", "model"
    ]).reset_index(drop=True)
    yearly_sort = [
        column for column in (
            "GAGE_ID", "threshold_quantile", "lead_days", "water_year"
        ) if column in yearly.columns
    ]
    yearly = yearly.sort_values(yearly_sort).reset_index(drop=True)
    if not exclusions.empty:
        exclusions["GAGE_ID"] = exclusions["GAGE_ID"].astype(str).str.zfill(8)
        exclusion_sort = [
            column for column in ("GAGE_ID", "lead_days", "threshold_name", "reason")
            if column in exclusions.columns
        ]
        exclusions = exclusions.sort_values(exclusion_sort).reset_index(drop=True)
    failures = sorted(failures, key=lambda item: str(item["GAGE_ID"]))
    logger.info("All basin models complete; starting spatial paired bootstrap")
    paired = paired_skill(metrics, cfg)
    logger.info("Spatial paired bootstrap complete; starting water-year block bootstrap")
    block_paired = water_year_block_skill(yearly, cfg, logger)
    logger.info("Water-year block bootstrap complete; starting Q5 severity contrasts")
    severity = severity_contrasts(yearly, cfg, logger)
    if block_paired.empty:
        raise RuntimeError("Water-year block bootstrap produced no eligible comparisons")
    summary = model_summary(metrics)
    basin_horizon = basin_horizons(metrics, cfg)
    basin_horizon_summary = summarize_basin_horizons(basin_horizon)
    lead_decisions, continental_horizon = continental_horizons(paired, cfg)
    block_lead_decisions, block_continental_horizon = continental_horizons(block_paired, cfg)

    attribute_columns = [
        column for column in [
            "GAGE_ID", "quality_tier", "is_reference", "AGGECOREGION", "CLASS",
            "area_km2", "log_area", "aridity", "BFI_AVE", "snow_fraction",
            "snowfall_fraction_of_pr", "recession_variance_exponent",
        ] if column in attributes
    ]
    horizon_enriched = basin_horizon.merge(
        attributes[attribute_columns], on="GAGE_ID", how="left", validate="many_to_one"
    )
    distribution_columns = [
        column for column in [
            "GAGE_ID", "standardized_excess_kurtosis", "standardized_tail_fraction",
            "student_t_delta_aic_vs_normal",
        ] if column in distribution
    ]
    horizon_enriched = horizon_enriched.merge(
        distribution[distribution_columns], on="GAGE_ID", how="left", validate="many_to_one"
    )
    primary_quantile = float(setting["primary_low_flow_quantile"])
    driver_population = horizon_enriched[
        horizon_enriched["threshold_quantile"].eq(primary_quantile)
        & horizon_enriched["reference_model"].eq("all_baselines")
    ].copy()
    predictors = [
        column for column in [
            "recession_variance_exponent", "standardized_excess_kurtosis",
            "standardized_tail_fraction", "BFI_AVE", "aridity", "snow_fraction", "log_area",
        ] if column in driver_population
    ]
    drivers = spearman_table(
        driver_population, ["predictability_horizon_days"], predictors
    ) if predictors else pd.DataFrame()

    scaling_decision = json.loads(
        (stage_dir(cfg, 16) / "SCALING_DECISION.json").read_text(encoding="utf-8")
    )
    distribution_decision = json.loads(
        (stage_dir(cfg, 17) / "DISTRIBUTION_DECISION.json").read_text(encoding="utf-8")
    )
    primary_horizon_row = continental_horizon[
        continental_horizon["threshold_quantile"].eq(primary_quantile)
        & continental_horizon["reference_model"].eq("all_baselines")
    ]
    block_primary_horizon_row = block_continental_horizon[
        block_continental_horizon["threshold_quantile"].eq(primary_quantile)
        & block_continental_horizon["reference_model"].eq("all_baselines")
    ]
    if primary_horizon_row.empty or block_primary_horizon_row.empty:
        raise RuntimeError("Primary all-baseline horizon was not estimable under both bootstraps")
    spatial_primary_horizon = int(primary_horizon_row["predictability_horizon_days"].iloc[0])
    block_primary_horizon = int(
        block_primary_horizon_row["predictability_horizon_days"].iloc[0]
    )
    primary_horizon = min(spatial_primary_horizon, block_primary_horizon)
    minimum_horizon = int(setting["minimum_scientifically_meaningful_horizon_days"])
    horizon_supported = bool(primary_horizon >= minimum_horizon)
    def positive_crps_leads(table: pd.DataFrame, model: str, reference: str) -> set[int]:
        selected = table[
            table["threshold_quantile"].eq(primary_quantile)
            & table["model"].eq(model)
            & table["reference_model"].eq(reference)
            & table["metric"].eq("crps")
        ]
        positive = (
            selected["bootstrap_ci_low"].gt(0)
            & selected["basins"].ge(int(setting["minimum_basins_per_lead_for_claim"]))
            & selected["spatial_groups"].ge(
                int(setting["minimum_spatial_groups_per_lead_for_claim"])
            )
        )
        return set(selected.loc[positive, "lead_days"].astype(int))

    non_gaussian_spatial_leads = positive_crps_leads(
        paired, MAIN_MODEL, STATE_GAUSSIAN_MODEL
    )
    non_gaussian_block_leads = positive_crps_leads(
        block_paired, MAIN_MODEL, STATE_GAUSSIAN_MODEL
    )
    non_gaussian_positive_leads = non_gaussian_spatial_leads & non_gaussian_block_leads
    non_gaussian_supported = bool(
        len(non_gaussian_positive_leads)
        >= int(setting["minimum_leads_with_non_gaussian_skill"])
    )
    state_spatial_leads = positive_crps_leads(
        paired, STATE_GAUSSIAN_MODEL, GAUSSIAN_MODEL
    )
    state_block_leads = positive_crps_leads(
        block_paired, STATE_GAUSSIAN_MODEL, GAUSSIAN_MODEL
    )
    state_positive_leads = state_spatial_leads & state_block_leads
    state_dependence_supported = bool(
        len(state_positive_leads) >= int(setting["minimum_leads_with_non_gaussian_skill"])
    )
    def positive_brier_leads(table: pd.DataFrame, model: str, reference: str) -> set[int]:
        selected = table[
            table["threshold_quantile"].eq(primary_quantile)
            & table["model"].eq(model)
            & table["reference_model"].eq(reference)
            & table["metric"].eq("brier")
        ]
        positive = (
            selected["bootstrap_ci_low"].gt(0)
            & selected["basins"].ge(int(setting["minimum_basins_per_lead_for_claim"]))
            & selected["spatial_groups"].ge(
                int(setting["minimum_spatial_groups_per_lead_for_claim"])
            )
        )
        return set(selected.loc[positive, "lead_days"].astype(int))

    hybrid_spatial_leads = positive_brier_leads(
        paired, OPERATIONAL_MODEL, MAIN_MODEL
    )
    hybrid_block_leads = positive_brier_leads(
        block_paired, OPERATIONAL_MODEL, MAIN_MODEL
    )
    hybrid_positive_leads = hybrid_spatial_leads & hybrid_block_leads
    hybrid_activation_lead = int(setting["recession_logistic_activation_lead_days"])
    hybrid_supported = bool(hybrid_activation_lead in hybrid_positive_leads)
    scientific = bool(
        scaling_decision["near_multiplicative_organizing_tendency_supported"]
        and distribution_decision["measurement_robust_scaling_supported"]
        and distribution_decision[
            "non_gaussian_measurement_and_rounding_robust_supported"
        ]
    )
    robust_horizon_lookup = {}
    for quantile in setting["low_flow_quantiles"]:
        quantile = float(quantile)
        spatial_row = continental_horizon[
            continental_horizon["threshold_quantile"].eq(quantile)
            & continental_horizon["reference_model"].eq("all_baselines")
        ]
        block_row = block_continental_horizon[
            block_continental_horizon["threshold_quantile"].eq(quantile)
            & block_continental_horizon["reference_model"].eq("all_baselines")
        ]
        if len(spatial_row) and len(block_row):
            robust_horizon_lookup[quantile] = min(
                int(spatial_row["predictability_horizon_days"].iloc[0]),
                int(block_row["predictability_horizon_days"].iloc[0]),
            )
    severity_order_supported = bool(
        len(robust_horizon_lookup) == len(setting["low_flow_quantiles"])
        and all(
            robust_horizon_lookup[left] >= robust_horizon_lookup[right]
            for left, right in zip(
                sorted(robust_horizon_lookup), sorted(robust_horizon_lookup)[1:]
            )
        )
    )
    severity_at_target = severity[
        severity["lead_days"].eq(minimum_horizon)
        & severity["basins"].ge(int(setting["minimum_basins_per_lead_for_claim"]))
        & severity["spatial_groups"].ge(
            int(setting["minimum_spatial_groups_per_lead_for_claim"])
        )
    ]
    severity_contrast_supported = bool(
        len(severity_at_target) == max(0, len(setting["low_flow_quantiles"]) - 1)
        and severity_at_target["bootstrap_ci_low"].gt(0).all()
    )
    severity_gradient_supported = bool(
        severity_order_supported and severity_contrast_supported
    )
    combined_skill_criteria_supported = bool(
        scientific and horizon_supported and non_gaussian_supported
        and state_dependence_supported and hybrid_supported
    )
    decision = {
        "core_dynamics_supported": scientific,
        "combined_skill_criteria_supported": combined_skill_criteria_supported,
        "predictability_horizon_supported": horizon_supported,
        "primary_threshold": threshold_name(primary_quantile),
        "primary_all_baseline_horizon_days": primary_horizon,
        "spatial_bootstrap_primary_all_baseline_horizon_days": spatial_primary_horizon,
        "water_year_block_primary_all_baseline_horizon_days": block_primary_horizon,
        "robust_all_baseline_horizons_by_threshold": {
            threshold_name(key): value for key, value in robust_horizon_lookup.items()
        },
        "minimum_scientifically_meaningful_horizon_days": minimum_horizon,
        "non_gaussian_incremental_value_supported": non_gaussian_supported,
        "non_gaussian_positive_crps_leads": sorted(non_gaussian_positive_leads),
        "state_dependence_incremental_value_supported": state_dependence_supported,
        "state_dependence_positive_crps_leads": sorted(state_positive_leads),
        "lead_adaptive_hybrid_incremental_value_supported": hybrid_supported,
        "lead_adaptive_hybrid_positive_brier_leads": sorted(hybrid_positive_leads),
        "operational_probability_model": OPERATIONAL_MODEL,
        "recession_logistic_activation_lead_days": hybrid_activation_lead,
        "recession_logistic_blend_weight": float(
            setting["recession_logistic_blend_weight"]
        ),
        "q5_horizon_not_shorter_than_q10_q20": severity_order_supported,
        "q5_skill_advantage_at_target_lead_supported": severity_contrast_supported,
        "threshold_severity_gradient_supported": severity_gradient_supported,
        "threshold_severity_method": (
            "Difference between ratios of observation-weighted aggregate Brier scores; "
            "water years, spatial groups, and basins are resampled hierarchically."
        ),
        "standardized_residual_collapse_supported": distribution_decision[
            "standardized_residual_collapse_supported"
        ],
        "state_specific_innovations_used": True,
        "minimum_basins_per_lead_for_claim": int(setting["minimum_basins_per_lead_for_claim"]),
        "minimum_spatial_groups_per_lead_for_claim": int(
            setting["minimum_spatial_groups_per_lead_for_claim"]
        ),
        "conditioning_statement": (
            "Skill applies to temporally held-out, observed precipitation- and proxy-snowmelt-screened "
            "dry-spell transitions. Future forcing is not a forecast input, so this is not an "
            "unconditional real-time streamflow forecast."
        ),
        "horizon_definition": (
            "Largest uninterrupted configured lead with positive basin-paired Brier improvement "
            "whose 95% confidence interval excludes zero under both the spatial hierarchy and a "
            "water-year block bootstrap; the all-baseline horizon must beat training climatology, "
            "persistence, and a training-only flow-bin probability model."
        ),
    }

    paths = {
        "metrics": out / "basin_predictability_metrics.csv.gz",
        "summary": out / "predictability_model_summary.csv",
        "paired": out / "paired_predictability_skill_summary.csv",
        "water_year_scores": out / "basin_water_year_skill_differences.csv.gz",
        "water_year_paired": out / "water_year_block_skill_summary.csv",
        "water_year_lead_decisions": out / "water_year_block_lead_skill_decisions.csv",
        "water_year_horizons": out / "water_year_block_predictability_horizons.csv",
        "severity_contrasts": out / "threshold_severity_skill_contrasts.csv",
        "lead_decisions": out / "continental_lead_skill_decisions.csv",
        "continental_horizons": out / "continental_predictability_horizons.csv",
        "basin_horizons": out / "basin_predictability_horizons.csv.gz",
        "basin_horizon_summary": out / "basin_predictability_horizon_summary.csv",
        "drivers": out / "predictability_horizon_driver_spearman.csv",
        "exclusions": out / "predictability_exclusions.csv.gz",
        "failures": out / "predictability_failures.csv",
        "analysis_summary_json": out / "ANALYSIS_SUMMARY.json",
        "analysis_summary_md": out / "ANALYSIS_SUMMARY.md",
    }
    metrics.to_csv(paths["metrics"], index=False, compression="gzip")
    summary.to_csv(paths["summary"], index=False)
    paired.to_csv(paths["paired"], index=False)
    yearly.to_csv(paths["water_year_scores"], index=False, compression="gzip")
    block_paired.to_csv(paths["water_year_paired"], index=False)
    block_lead_decisions.to_csv(paths["water_year_lead_decisions"], index=False)
    block_continental_horizon.to_csv(paths["water_year_horizons"], index=False)
    severity.to_csv(paths["severity_contrasts"], index=False)
    lead_decisions.to_csv(paths["lead_decisions"], index=False)
    continental_horizon.to_csv(paths["continental_horizons"], index=False)
    horizon_enriched.to_csv(paths["basin_horizons"], index=False, compression="gzip")
    basin_horizon_summary.to_csv(paths["basin_horizon_summary"], index=False)
    drivers.to_csv(paths["drivers"], index=False)
    exclusions.to_csv(paths["exclusions"], index=False, compression="gzip")
    pd.DataFrame(failures, columns=["GAGE_ID", "error"]).to_csv(paths["failures"], index=False)
    paths["analysis_summary_json"].write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    write_analysis_summary(paths["analysis_summary_md"], decision)
    if failures:
        raise RuntimeError(
            f"{len(failures)} unexpected Stage-18 failures; inspect {paths['failures']}"
        )
    write_receipt(cfg, STAGE, [
        *paths.values(), by_basin, by_basin_exclusions, by_basin_water_year,
    ], {
        "basins_requested": len(tasks),
        "basins_analyzed": metrics["GAGE_ID"].nunique(),
        "thresholds": metrics["threshold_name"].nunique(),
        "maximum_lead_days": int(metrics["lead_days"].max()),
        "model_definitions": len(ALL_MODELS),
        "core_dynamics_supported": scientific,
        "primary_all_baseline_horizon_days": primary_horizon,
        "predictability_horizon_supported": horizon_supported,
        "combined_skill_criteria_supported": combined_skill_criteria_supported,
    })
    logger.info(
        "Stage 18 complete: core dynamics=%s; Q10 all-baseline horizon=%d days; horizon criterion=%s",
        scientific, primary_horizon, horizon_supported,
    )


if __name__ == "__main__":
    main()
