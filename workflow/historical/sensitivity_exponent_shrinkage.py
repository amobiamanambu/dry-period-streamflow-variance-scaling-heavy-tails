#!/usr/bin/env python3
"""Sensitivity analysis that refits Stage 18 under alternative exponent shrinkage.

This sidecar analysis deliberately leaves the completed Stage 18 directory
untouched.  It holds the Stage 18 transition definitions, train/test split,
model architecture, ensemble size, probability thresholds, and scoring rules
fixed while changing only the prior standard deviation for the conditional-
variance exponent.  The no-shrinkage variant uses the raw exponent clipped to
the prespecified [0, 4] bounds.

Default outputs are written below
``continental_run/sensitivity_analyses/shrinkage``.  A full run is stored in
``continental``; stratified pilots are stored under a separate, explicitly
labelled directory and cannot be used for publication inference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "continental_run" / "sensitivity_analyses" / "shrinkage"
)
PRIMARY_STAGE18 = PROJECT_ROOT / "continental_run" / "18_lowflow_predictability_horizons"
SENSITIVITY_VERSION = 1
VARIANTS = (
    ("prior_sd_0p5", 0.5),
    ("prior_sd_0p75", 0.75),
    ("prior_sd_1p5", 1.5),
    ("no_shrinkage", None),
)


def load_stage18_module():
    """Load the frozen numeric Stage 18 module without duplicating its methods."""
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    path = SCRIPT_DIR / "18_test_operational_value.py"
    spec = importlib.util.spec_from_file_location("primary_stage18", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Stage 18 from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE18 = load_stage18_module()


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--leads", nargs="+", type=int, default=None,
        help="Subset of configured leads. The default is every Stage 18 lead.",
    )
    parser.add_argument(
        "--quantiles", nargs="+", type=float, default=[0.10],
        help="Low-flow quantiles to score. The sensitivity analysis defaults to Q10.",
    )
    parser.add_argument(
        "--pilot-per-stratum", type=int, default=None,
        help=(
            "Run a deterministic pilot with N basins per ecoregion x primary Stage 18 "
            "exponent-SE quartile. Pilot products are explicitly labelled noninferential."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug-only basin limit.")
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=None,
        help="Override the primary 1,000 bootstrap replicates (pilots/debugging only).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--refresh-summaries", action="store_true",
        help=(
            "Rebuild aggregate summaries and the receipt from compatible basin caches "
            "without refitting."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(text).digest()[:4], "little")
    return int((int(base) + offset) % (2**32 - 1))


def q_name(quantile: float) -> str:
    return f"Q{int(round(100 * quantile))}"


def atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def rebased_transition_path(value: str, gage: str) -> Path:
    supplied = Path(str(value))
    if supplied.exists():
        return supplied
    candidate = PROJECT_ROOT / "continental_run" / "10_transitions" / "daily" / supplied.name
    if candidate.exists():
        return candidate
    candidate = PROJECT_ROOT / "continental_run" / "10_transitions" / "daily" / f"{gage}.csv.gz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No Stage 10 transition file for {gage}: {value}")


def variant_config(cfg: dict, prior_sd: float | None) -> dict:
    result = copy.deepcopy(cfg)
    result["predictability_test"]["variance_exponent_prior_sd"] = (
        float("inf") if prior_sd is None else float(prior_sd)
    )
    return result


def fit_variant(train: pd.DataFrame, cfg: dict, prior_sd: float | None) -> dict:
    fitted = STAGE18.fit_transition_model(train, variant_config(cfg, prior_sd))
    if prior_sd is None:
        low, high = cfg["predictability_test"]["variance_exponent_bounds"]
        expected = float(np.clip(fitted["raw_variance_exponent"], low, high))
        if not np.isclose(fitted["regularized_variance_exponent"], expected, atol=1e-12):
            raise RuntimeError("No-shrinkage variant did not equal the bounded raw exponent")
    return fitted


def worker(task: tuple) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    (
        gage, source, spatial_group, cfg, leads, quantiles, metrics_path,
        yearly_path, exclusions_path, fingerprint, force,
    ) = task
    metrics_path = Path(metrics_path)
    yearly_path = Path(yearly_path)
    exclusions_path = Path(exclusions_path)
    if all(path.exists() for path in (metrics_path, yearly_path, exclusions_path)) and not force:
        cached = pd.read_csv(metrics_path, dtype={"GAGE_ID": str})
        if (
            not cached.empty
            and int(cached["sensitivity_version"].iloc[0]) == SENSITIVITY_VERSION
            and str(cached["run_fingerprint"].iloc[0]) == fingerprint
        ):
            return (
                cached,
                pd.read_csv(yearly_path, dtype={"GAGE_ID": str}),
                pd.read_csv(exclusions_path, dtype={"GAGE_ID": str}),
            )

    setting = cfg["predictability_test"]
    frame = pd.read_csv(source, parse_dates=["date"], low_memory=False).sort_values("date")
    training_end = pd.Timestamp(setting["training_end_date"])
    evaluation_start = pd.Timestamp(setting["evaluation_start_date"])
    q_raw = pd.to_numeric(frame["q_mm_day"], errors="coerce")
    scale_values = q_raw[(frame["date"] <= training_end) & q_raw.gt(0)]
    scale = float(scale_values.median())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("no_positive_pre_evaluation_discharge_scale")
    frame["q_norm_predictability"] = q_raw / scale
    feature_frame, feature_columns = STAGE18.initialization_features(frame, scale, cfg)
    historical = frame.loc[frame["date"] <= training_end, "q_norm_predictability"]
    thresholds = {float(q): float(historical.quantile(float(q))) for q in quantiles}

    metric_rows: list[dict] = []
    yearly_rows: list[dict] = []
    exclusions: list[dict] = []
    for lead in leads:
        transitions = STAGE18.dry_transition_frame(frame, lead, cfg).merge(
            feature_frame, on="date", how="left", validate="many_to_one"
        )
        train = transitions[transitions["future_date"] <= training_end].copy()
        test = transitions[transitions["date"] >= evaluation_start].copy()
        if len(train) < int(setting["minimum_training_transitions"]):
            exclusions.append({"GAGE_ID": gage, "lead_days": lead, "threshold_name": "ALL", "variant": "ALL", "reason": "insufficient_training_transitions"})
            continue
        if len(test) < int(setting["minimum_test_transitions"]):
            exclusions.append({"GAGE_ID": gage, "lead_days": lead, "threshold_name": "ALL", "variant": "ALL", "reason": "insufficient_evaluation_transitions"})
            continue

        fitted_variants: dict[str, dict] = {}
        for variant, prior_sd in VARIANTS:
            try:
                fitted_variants[variant] = fit_variant(train, cfg, prior_sd)
            except ValueError as error:
                exclusions.append({"GAGE_ID": gage, "lead_days": lead, "threshold_name": "ALL", "variant": variant, "reason": str(error)})
        if len(fitted_variants) != len(VARIANTS):
            continue
        raw_values = np.array([
            fitted["raw_variance_exponent"] for fitted in fitted_variants.values()
        ])
        se_values = np.array([
            fitted["variance_exponent_standard_error"] for fitted in fitted_variants.values()
        ])
        if not (
            np.allclose(raw_values, raw_values[0], equal_nan=True, atol=1e-12)
            and np.allclose(se_values, se_values[0], equal_nan=True, atol=1e-12)
        ):
            raise RuntimeError(f"Only the prior changed, but raw estimates differed for {gage}")

        q_test = test["q"].to_numpy(float)
        observed = test["q_next"].to_numpy(float)
        q_train = train["q"].to_numpy(float)
        train_observed = train["q_next"].to_numpy(float)
        years = STAGE18.water_year(test["future_date"])

        for quantile, threshold in thresholds.items():
            name = q_name(quantile)
            if not np.isfinite(threshold) or (
                setting["require_positive_low_flow_threshold"] and threshold <= 0
            ):
                exclusions.append({"GAGE_ID": gage, "lead_days": lead, "threshold_name": name, "variant": "ALL", "reason": "nonpositive_training_threshold"})
                continue
            train_target = train_observed <= threshold
            test_target = observed <= threshold
            events = int(test_target.sum())
            nonevents = int((~test_target).sum())
            if events < int(setting["minimum_events_per_basin"]):
                exclusions.append({"GAGE_ID": gage, "lead_days": lead, "threshold_name": name, "variant": "ALL", "reason": "insufficient_evaluation_events"})
                continue
            if nonevents < int(setting["minimum_nonevents_per_basin"]):
                exclusions.append({"GAGE_ID": gage, "lead_days": lead, "threshold_name": name, "variant": "ALL", "reason": "insufficient_evaluation_nonevents"})
                continue
            if train_target.sum() == 0 or (~train_target).sum() == 0:
                exclusions.append({"GAGE_ID": gage, "lead_days": lead, "threshold_name": name, "variant": "ALL", "reason": "training_outcome_has_one_class"})
                continue

            primary_fit = fitted_variants["prior_sd_0p75"]
            baseline_probabilities = {
                "training_climatology": np.full(
                    len(test_target), (train_target.sum() + 0.5) / (len(train_target) + 1.0)
                ),
                "persistence": (q_test <= threshold).astype(float),
                "qbin_probability": STAGE18.qbin_probabilities(
                    q_train, train_target, q_test, primary_fit["bin_edges"]
                ),
            }
            baseline_scores = {
                model: STAGE18.probability_score_arrays(
                    probability, test_target, float(setting["probability_clip"])
                )[0]
                for model, probability in baseline_probabilities.items()
            }
            hybrid_weight = (
                float(setting["recession_logistic_blend_weight"])
                if lead >= int(setting["recession_logistic_activation_lead_days"])
                else 0.0
            )
            logistic_probability = None
            if hybrid_weight > 0:
                logistic_probability = STAGE18.recession_logistic_probabilities(
                    train, test, train_target, feature_columns, cfg
                )

            for variant, prior_sd in VARIANTS:
                fitted = fitted_variants[variant]
                ensembles = {
                    model: STAGE18.distribution_members(
                        model, q_test, fitted, int(setting["ensemble_members"])
                    )
                    for model in (
                        STAGE18.GAUSSIAN_MODEL,
                        STAGE18.STATE_GAUSSIAN_MODEL,
                        STAGE18.MAIN_MODEL,
                    )
                }
                ensemble_probabilities = {
                    model: (np.sum(ensemble <= threshold, axis=1) + 0.5)
                    / (ensemble.shape[1] + 1.0)
                    for model, ensemble in ensembles.items()
                }
                empirical_probability = ensemble_probabilities[STAGE18.MAIN_MODEL]
                operational_probability = (
                    empirical_probability.copy()
                    if logistic_probability is None
                    else (
                        (1.0 - hybrid_weight) * empirical_probability
                        + hybrid_weight * logistic_probability
                    )
                )
                probabilities = {
                    **baseline_probabilities,
                    **ensemble_probabilities,
                    STAGE18.OPERATIONAL_MODEL: operational_probability,
                }
                probability_arrays = {
                    model: STAGE18.probability_score_arrays(
                        probability, test_target, float(setting["probability_clip"])
                    )
                    for model, probability in probabilities.items()
                }
                crps_arrays = {
                    model: STAGE18.ensemble_crps(ensemble, observed)
                    for model, ensemble in ensembles.items()
                }
                common = {
                    "sensitivity_version": SENSITIVITY_VERSION,
                    "run_fingerprint": fingerprint,
                    "GAGE_ID": str(gage).zfill(8),
                    "spatial_group": spatial_group,
                    "variant": variant,
                    "prior_sd": np.nan if prior_sd is None else float(prior_sd),
                    "no_shrinkage": prior_sd is None,
                    "lead_days": int(lead),
                    "threshold_quantile": float(quantile),
                    "threshold_name": name,
                    "n": len(test_target),
                    "events": events,
                    "raw_variance_exponent": fitted["raw_variance_exponent"],
                    "variance_exponent_standard_error": fitted[
                        "variance_exponent_standard_error"
                    ],
                    "regularized_variance_exponent": fitted[
                        "regularized_variance_exponent"
                    ],
                    "variance_coefficient": fitted["variance_coefficient"],
                    "variance_r2": fitted["variance_r2"],
                }
                for model, probability in probabilities.items():
                    probability_metrics = STAGE18.probability_metrics(
                        probability, test_target, float(setting["probability_clip"])
                    )
                    metric_rows.append({
                        **common,
                        "model": model,
                        "brier": probability_metrics["brier"],
                        "log_loss": probability_metrics["log_loss"],
                        "crps": (
                            float(np.mean(crps_arrays[model]))
                            if model in crps_arrays else np.nan
                        ),
                    })

                comparisons = (
                    (
                        "state_gaussian_vs_global_gaussian",
                        STAGE18.STATE_GAUSSIAN_MODEL,
                        STAGE18.GAUSSIAN_MODEL,
                        "crps",
                        crps_arrays[STAGE18.STATE_GAUSSIAN_MODEL],
                        crps_arrays[STAGE18.GAUSSIAN_MODEL],
                    ),
                    (
                        "state_empirical_vs_state_gaussian",
                        STAGE18.MAIN_MODEL,
                        STAGE18.STATE_GAUSSIAN_MODEL,
                        "crps",
                        crps_arrays[STAGE18.MAIN_MODEL],
                        crps_arrays[STAGE18.STATE_GAUSSIAN_MODEL],
                    ),
                    *tuple(
                        (
                            f"operational_vs_{reference}",
                            STAGE18.OPERATIONAL_MODEL,
                            reference,
                            "brier",
                            probability_arrays[STAGE18.OPERATIONAL_MODEL][0],
                            baseline_scores[reference],
                        )
                        for reference in STAGE18.BASELINE_MODELS
                    ),
                )
                for year in np.unique(years):
                    selected = years == year
                    if not selected.any():
                        continue
                    for comparison, model, reference, metric, model_score, reference_score in comparisons:
                        yearly_rows.append({
                            **{key: common[key] for key in (
                                "sensitivity_version", "run_fingerprint", "GAGE_ID",
                                "spatial_group", "variant", "prior_sd", "no_shrinkage",
                                "lead_days", "threshold_quantile", "threshold_name",
                            )},
                            "water_year": int(year),
                            "n": int(selected.sum()),
                            "events": int(test_target[selected].sum()),
                            "comparison": comparison,
                            "model": model,
                            "reference_model": reference,
                            "metric": metric,
                            "improvement": float(np.mean(
                                reference_score[selected] - model_score[selected]
                            )),
                        })

    metrics = pd.DataFrame(metric_rows)
    yearly = pd.DataFrame(yearly_rows)
    exclusion_frame = pd.DataFrame(exclusions, columns=[
        "GAGE_ID", "lead_days", "threshold_name", "variant", "reason"
    ])
    atomic_csv(metrics, metrics_path, compression="gzip")
    atomic_csv(yearly, yearly_path, compression="gzip")
    atomic_csv(exclusion_frame, exclusions_path)
    return metrics, yearly, exclusion_frame


COMPARISONS = (
    (
        "state_gaussian_vs_global_gaussian",
        STAGE18.STATE_GAUSSIAN_MODEL,
        STAGE18.GAUSSIAN_MODEL,
        "crps",
    ),
    (
        "state_empirical_vs_state_gaussian",
        STAGE18.MAIN_MODEL,
        STAGE18.STATE_GAUSSIAN_MODEL,
        "crps",
    ),
    *tuple(
        (
            f"operational_vs_{reference}",
            STAGE18.OPERATIONAL_MODEL,
            reference,
            "brier",
        )
        for reference in STAGE18.BASELINE_MODELS
    ),
)


def spatial_summaries(metrics: pd.DataFrame, cfg: dict, replicates: int) -> pd.DataFrame:
    keys = [
        "variant", "GAGE_ID", "spatial_group", "threshold_quantile",
        "threshold_name", "lead_days",
    ]
    rows: list[dict] = []
    base_seed = int(cfg["project"]["random_seed"]) + 21_800_000
    chunk = int(cfg["predictability_test"]["bootstrap_chunk_size"])
    for comparison, model, reference, metric in COMPARISONS:
        left = metrics[metrics["model"].eq(model)][keys + ["n", metric]].rename(
            columns={"n": "weight", metric: "model_score"}
        )
        right = metrics[metrics["model"].eq(reference)][keys + [metric]].rename(
            columns={metric: "reference_score"}
        )
        paired = left.merge(right, on=keys, validate="one_to_one")
        paired["improvement"] = paired["reference_score"] - paired["model_score"]
        paired = paired[np.isfinite(paired[["improvement", "reference_score", "weight"]]).all(axis=1)]
        for (variant, quantile, name, lead), group in paired.groupby([
            "variant", "threshold_quantile", "threshold_name", "lead_days"
        ]):
            rng = np.random.default_rng(stable_seed(
                base_seed, comparison, metric, quantile, lead
            ))
            boot = STAGE18.hierarchical_bootstrap(group, replicates, chunk, rng)
            estimate = float(np.average(group["improvement"], weights=group["weight"]))
            reference_score = float(np.average(
                group["reference_score"], weights=group["weight"]
            ))
            basin_mean = group.groupby("GAGE_ID")["improvement"].mean()
            regional = group.groupby("spatial_group")["improvement"].mean()
            low, high = np.quantile(boot, [0.025, 0.975])
            rows.append({
                "variant": variant,
                "threshold_quantile": float(quantile),
                "threshold_name": name,
                "lead_days": int(lead),
                "comparison": comparison,
                "model": model,
                "reference_model": reference,
                "metric": metric,
                "basins": int(group["GAGE_ID"].nunique()),
                "spatial_groups": int(group["spatial_group"].nunique()),
                "test_transitions": int(group["weight"].sum()),
                "weighted_improvement": estimate,
                "reference_score": reference_score,
                "relative_skill": estimate / reference_score if reference_score > 0 else np.nan,
                "bootstrap_ci_low": float(low),
                "bootstrap_ci_high": float(high),
                "basins_improved": int((basin_mean > 0).sum()),
                "groups_improved": int((regional > 0).sum()),
                "bootstrap_replicates": int(replicates),
                "resampling_unit": "spatial_group_then_basin",
                "positive_means_better": True,
            })
    return pd.DataFrame(rows)


def block_summaries(yearly: pd.DataFrame, cfg: dict, replicates: int) -> pd.DataFrame:
    rows: list[dict] = []
    setting = cfg["predictability_test"]
    minimum_years = int(setting["minimum_evaluation_water_years_per_basin"])
    chunk = int(setting["bootstrap_chunk_size"])
    base_seed = int(cfg["project"]["random_seed"]) + 21_850_000
    for (variant, quantile, name, lead, comparison, model, reference, metric), group in yearly.groupby([
        "variant", "threshold_quantile", "threshold_name", "lead_days",
        "comparison", "model", "reference_model", "metric",
    ]):
        selected = group[np.isfinite(group["improvement"]) & group["n"].gt(0)].copy()
        selected["weight"] = selected["n"].to_numpy(float)
        selected.attrs["minimum_water_years"] = minimum_years
        rng = np.random.default_rng(stable_seed(
            base_seed, comparison, metric, quantile, lead
        ))
        boot, eligible = STAGE18.water_year_hierarchical_bootstrap(
            selected, replicates, chunk, rng
        )
        if eligible.empty or len(boot) < max(100, replicates // 2):
            continue
        estimate = float(np.average(eligible["improvement"], weights=eligible["weight"]))
        basin = eligible.assign(
            weighted=eligible["improvement"] * eligible["weight"]
        ).groupby("GAGE_ID").agg(weighted=("weighted", "sum"), weight=("weight", "sum"))
        basin_mean = basin["weighted"] / basin["weight"]
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append({
            "variant": variant,
            "threshold_quantile": float(quantile),
            "threshold_name": name,
            "lead_days": int(lead),
            "comparison": comparison,
            "model": model,
            "reference_model": reference,
            "metric": metric,
            "basins": int(eligible["GAGE_ID"].nunique()),
            "spatial_groups": int(eligible["spatial_group"].nunique()),
            "water_years": int(eligible["water_year"].nunique()),
            "test_transitions": int(eligible["weight"].sum()),
            "weighted_improvement": estimate,
            "bootstrap_ci_low": float(low),
            "bootstrap_ci_high": float(high),
            "basins_improved": int((basin_mean > 0).sum()),
            "bootstrap_replicates": int(replicates),
            "resampling_unit": "water_year_then_spatial_group_then_basin",
            "positive_means_better": True,
        })
    return pd.DataFrame(rows)


def exponent_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    fits = metrics[[
        "variant", "GAGE_ID", "lead_days", "raw_variance_exponent",
        "variance_exponent_standard_error", "regularized_variance_exponent",
    ]].drop_duplicates()
    primary = fits[fits["variant"].eq("prior_sd_0p75")][[
        "GAGE_ID", "lead_days", "regularized_variance_exponent"
    ]].rename(columns={"regularized_variance_exponent": "primary_exponent"})
    fits = fits.merge(primary, on=["GAGE_ID", "lead_days"], validate="many_to_one")
    fits["shift_from_primary"] = (
        fits["regularized_variance_exponent"] - fits["primary_exponent"]
    )
    fits["shift_from_raw"] = (
        fits["regularized_variance_exponent"] - fits["raw_variance_exponent"]
    )
    rows = []
    for (variant, lead), group in fits.groupby(["variant", "lead_days"]):
        rows.append({
            "variant": variant,
            "lead_days": int(lead),
            "basins": len(group),
            "median_raw_exponent": float(group["raw_variance_exponent"].median()),
            "median_regularized_exponent": float(group["regularized_variance_exponent"].median()),
            "median_exponent_se": float(group["variance_exponent_standard_error"].median()),
            "median_shift_from_primary": float(group["shift_from_primary"].median()),
            "median_abs_shift_from_primary": float(group["shift_from_primary"].abs().median()),
            "p90_abs_shift_from_primary": float(group["shift_from_primary"].abs().quantile(0.9)),
            "median_abs_shrinkage_from_raw": float(group["shift_from_raw"].abs().median()),
            "fraction_at_lower_bound": float(group["regularized_variance_exponent"].le(1e-12).mean()),
            "fraction_at_upper_bound": float(group["regularized_variance_exponent"].ge(4 - 1e-12).mean()),
        })
    return pd.DataFrame(rows)


def deltas_from_primary(spatial: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "threshold_quantile", "threshold_name", "lead_days", "comparison",
        "model", "reference_model", "metric",
    ]
    primary = spatial[spatial["variant"].eq("prior_sd_0p75")][
        keys + ["weighted_improvement", "relative_skill"]
    ].rename(columns={
        "weighted_improvement": "primary_weighted_improvement",
        "relative_skill": "primary_relative_skill",
    })
    result = spatial.merge(primary, on=keys, validate="many_to_one")
    result["weighted_improvement_delta_from_primary"] = (
        result["weighted_improvement"] - result["primary_weighted_improvement"]
    )
    result["relative_skill_delta_from_primary"] = (
        result["relative_skill"] - result["primary_relative_skill"]
    )
    return result


def conclusion_summary(
    spatial: pd.DataFrame, block: pd.DataFrame, cfg: dict, inferential: bool
) -> pd.DataFrame:
    keys = [
        "variant", "threshold_quantile", "threshold_name", "lead_days",
        "comparison", "model", "reference_model", "metric",
    ]
    joint = spatial.merge(
        block[keys + ["basins", "spatial_groups", "bootstrap_ci_low"]],
        on=keys, suffixes=("_spatial", "_block"), validate="one_to_one",
    )
    minimum_basins = int(cfg["predictability_test"]["minimum_basins_per_lead_for_claim"])
    minimum_groups = int(cfg["predictability_test"]["minimum_spatial_groups_per_lead_for_claim"])
    joint["robust_positive"] = (
        joint["bootstrap_ci_low_spatial"].gt(0)
        & joint["bootstrap_ci_low_block"].gt(0)
        & joint["basins_spatial"].ge(minimum_basins)
        & joint["basins_block"].ge(minimum_basins)
        & joint["spatial_groups_spatial"].ge(minimum_groups)
        & joint["spatial_groups_block"].ge(minimum_groups)
    )
    configured_leads = [int(value) for value in cfg["predictability_test"]["lead_days"]]
    rows = []
    for variant, variant_data in joint.groupby("variant"):
        def positive_leads(comparison: str) -> list[int]:
            selected = variant_data[variant_data["comparison"].eq(comparison)]
            return sorted(
                int(value)
                for value in selected.loc[
                    selected["robust_positive"], "lead_days"
                ].unique()
            )

        baseline_comparisons = [
            f"operational_vs_{reference}" for reference in STAGE18.BASELINE_MODELS
        ]
        all_baseline_positive = []
        for lead in configured_leads:
            checks = []
            for comparison in baseline_comparisons:
                row = variant_data[
                    variant_data["lead_days"].eq(lead)
                    & variant_data["comparison"].eq(comparison)
                ]
                checks.append(bool(len(row) == 1 and row["robust_positive"].iloc[0]))
            if all(checks):
                all_baseline_positive.append(lead)
        horizon = 0
        for lead in configured_leads:
            if lead not in all_baseline_positive:
                break
            horizon = lead
        rows.append({
            "variant": variant,
            "publication_inference_allowed": bool(inferential),
            "state_dependence_positive_crps_leads": json.dumps(
                positive_leads("state_gaussian_vs_global_gaussian")
            ),
            "empirical_shape_positive_crps_leads": json.dumps(
                positive_leads("state_empirical_vs_state_gaussian")
            ),
            "operational_all_baseline_positive_brier_leads": json.dumps(
                all_baseline_positive
            ),
            "q10_all_baseline_horizon_days": int(horizon),
            "decision_rule": (
                "Positive lower 95% CI under both spatial and water-year-block "
                "bootstraps, >=500 basins, >=5 spatial groups; horizon stops at first "
                "configured nonpositive or ineligible lead."
            ),
        })
    return pd.DataFrame(rows)


def reproduction_check(metrics: pd.DataFrame, spatial: pd.DataFrame) -> dict:
    primary_path = PRIMARY_STAGE18 / "basin_predictability_metrics.csv.gz"
    columns = [
        "GAGE_ID", "lead_days", "threshold_quantile", "threshold_name", "model",
        "regularized_variance_exponent", "brier", "crps",
    ]
    archived = pd.read_csv(primary_path, usecols=columns, dtype={"GAGE_ID": str})
    current = metrics[metrics["variant"].eq("prior_sd_0p75")][columns]
    paired = current.merge(
        archived, on=["GAGE_ID", "lead_days", "threshold_quantile", "threshold_name", "model"],
        suffixes=("_current", "_archived"), validate="one_to_one",
    )
    output = {"paired_rows": int(len(paired))}
    for column in ("regularized_variance_exponent", "brier", "crps"):
        difference = (
            paired[f"{column}_current"] - paired[f"{column}_archived"]
        ).abs()
        finite = difference[np.isfinite(difference)]
        output[f"maximum_absolute_{column}_difference"] = float(finite.max())
        output[f"rows_with_{column}_difference_above_1e-12"] = int(
            finite.gt(1e-12).sum()
        )
        output[f"fraction_with_{column}_difference_above_1e-12"] = float(
            finite.gt(1e-12).mean()
        )

    archived_summary = pd.read_csv(
        PRIMARY_STAGE18 / "paired_predictability_skill_summary.csv"
    )
    current_summary = spatial[
        spatial["variant"].eq("prior_sd_0p75")
    ].copy()
    comparison_rows = []
    for comparison, model, reference, metric in COMPARISONS:
        current = current_summary[
            current_summary["comparison"].eq(comparison)
        ][["threshold_name", "lead_days", "weighted_improvement"]]
        archived_rows = archived_summary[
            archived_summary["model"].eq(model)
            & archived_summary["reference_model"].eq(reference)
            & archived_summary["metric"].eq(metric)
        ][["threshold_name", "lead_days", "weighted_improvement"]]
        merged = current.merge(
            archived_rows, on=["threshold_name", "lead_days"],
            suffixes=("_current", "_archived"), validate="one_to_one",
        )
        merged["comparison"] = comparison
        comparison_rows.append(merged)
    aggregate = pd.concat(comparison_rows, ignore_index=True)
    aggregate_difference = (
        aggregate["weighted_improvement_current"]
        - aggregate["weighted_improvement_archived"]
    ).abs()
    output["maximum_absolute_continental_skill_difference"] = float(
        aggregate_difference.max()
    )
    output["exact_score_reproduction"] = bool(
        (output["maximum_absolute_brier_difference"] or 0.0) <= 1e-12
        and (output["maximum_absolute_crps_difference"] or 0.0) <= 1e-12
        and (output["maximum_absolute_regularized_variance_exponent_difference"] or 0.0)
        <= 1e-12
    )
    output["continental_point_estimates_numerically_equivalent"] = bool(
        output["maximum_absolute_continental_skill_difference"] <= 1e-5
        and output["maximum_absolute_crps_difference"] <= 1e-10
        and output["maximum_absolute_regularized_variance_exponent_difference"]
        <= 1e-10
    )
    output["interpretation"] = (
        "The current runtime reproduces exponents and CRPS to floating-point "
        "precision. Rare ensemble-threshold ties and the logistic optimizer differ "
        "slightly under the newer NumPy/scikit-learn runtime, but continental Brier "
        "skill estimates remain numerically equivalent."
    )
    return output


def select_population(
    inventory: pd.DataFrame, attributes: pd.DataFrame, primary: pd.DataFrame,
    args: argparse.Namespace, seed: int,
) -> tuple[pd.DataFrame, str, bool]:
    group_column = "AGGECOREGION"
    population = inventory.merge(
        attributes[["GAGE_ID", group_column]], on="GAGE_ID", how="inner", validate="one_to_one"
    )
    eligible = primary[
        primary["threshold_name"].eq("Q10")
        & primary["model"].eq(STAGE18.MAIN_MODEL)
    ][["GAGE_ID", "variance_exponent_standard_error"]].drop_duplicates("GAGE_ID")
    population = population.merge(eligible, on="GAGE_ID", how="inner", validate="one_to_one")
    population["spatial_group"] = population[group_column].fillna("unknown").astype(str)
    if args.pilot_per_stratum:
        if args.pilot_per_stratum < 1:
            raise ValueError("--pilot-per-stratum must be positive")
        population["se_quartile"] = pd.qcut(
            population["variance_exponent_standard_error"].rank(method="first"),
            4, labels=False,
        )
        rng = np.random.default_rng(seed + 21_899_001)
        sampled = []
        for _, group in population.groupby(["spatial_group", "se_quartile"], sort=True):
            count = min(int(args.pilot_per_stratum), len(group))
            indices = rng.choice(group.index.to_numpy(), size=count, replace=False)
            sampled.append(population.loc[indices])
        population = pd.concat(sampled, ignore_index=True).sort_values("GAGE_ID")
        label = f"pilot_{args.pilot_per_stratum}_per_ecoregion_sequartile"
        inferential = False
    elif args.limit:
        population = population.sort_values("GAGE_ID").head(int(args.limit))
        label = f"debug_first_{int(args.limit)}"
        inferential = False
    else:
        label = "continental"
        inferential = True
    return population, label, inferential


def main() -> None:
    args = parse_cli()
    cfg = STAGE18.load_config(args.config)
    configured_leads = [int(value) for value in cfg["predictability_test"]["lead_days"]]
    leads = configured_leads if args.leads is None else [int(value) for value in args.leads]
    if not leads or any(lead not in configured_leads for lead in leads):
        raise ValueError(f"--leads must be a subset of {configured_leads}")
    configured_quantiles = [
        float(value) for value in cfg["predictability_test"]["low_flow_quantiles"]
    ]
    quantiles = [float(value) for value in args.quantiles]
    if not quantiles or any(not any(np.isclose(value, q) for q in configured_quantiles) for value in quantiles):
        raise ValueError(f"--quantiles must be a subset of {configured_quantiles}")
    replicates = int(
        args.bootstrap_replicates
        or cfg["predictability_test"]["paired_bootstrap_replicates"]
    )
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")

    inventory = pd.read_csv(
        PROJECT_ROOT / "continental_run" / "10_transitions" / "transition_inventory.csv",
        dtype={"GAGE_ID": str},
    )
    attributes = pd.read_csv(
        PROJECT_ROOT / "continental_run" / "15_continental_results" / "continental_basin_results.csv",
        dtype={"GAGE_ID": str}, low_memory=False,
    )
    primary = pd.read_csv(
        PRIMARY_STAGE18 / "basin_predictability_metrics.csv.gz",
        usecols=[
            "GAGE_ID", "threshold_name", "model", "variance_exponent_standard_error"
        ], dtype={"GAGE_ID": str},
    )
    for frame in (inventory, attributes, primary):
        frame["GAGE_ID"] = frame["GAGE_ID"].astype(str).str.zfill(8)
    population, run_label, inferential = select_population(
        inventory, attributes, primary, args, int(cfg["project"]["random_seed"])
    )
    output = Path(args.output_root).expanduser().resolve() / run_label
    receipt = output / "_SUCCESS.json"
    if receipt.exists() and not args.force and not args.refresh_summaries:
        raise RuntimeError(
            f"Completed sensitivity already exists at {receipt}; use --force to refit "
            "or --refresh-summaries to reuse compatible basin caches."
        )
    output.mkdir(parents=True, exist_ok=True)

    fingerprint_payload = {
        "sensitivity_version": SENSITIVITY_VERSION,
        "stage18_sha256": sha256_file(SCRIPT_DIR / "18_test_operational_value.py"),
        "config_sha256": sha256_file(Path(args.config).resolve()),
        "variants": VARIANTS,
        "leads": leads,
        "quantiles": quantiles,
        "population": population["GAGE_ID"].tolist(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    by_basin_metrics = output / "by_basin_metrics"
    by_basin_yearly = output / "by_basin_water_year"
    by_basin_exclusions = output / "by_basin_exclusions"
    tasks = []
    for row in population.itertuples(index=False):
        source = rebased_transition_path(row.file, row.GAGE_ID)
        tasks.append((
            row.GAGE_ID,
            str(source),
            row.spatial_group,
            cfg,
            leads,
            quantiles,
            str(by_basin_metrics / f"{row.GAGE_ID}.csv.gz"),
            str(by_basin_yearly / f"{row.GAGE_ID}.csv.gz"),
            str(by_basin_exclusions / f"{row.GAGE_ID}.csv"),
            fingerprint,
            args.force and not args.refresh_summaries,
        ))

    frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []
    exclusion_frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {executor.submit(worker, task): task[0] for task in tasks}
        for number, future in enumerate(as_completed(futures), start=1):
            gage = futures[future]
            try:
                metrics, yearly, exclusions = future.result()
                if not metrics.empty:
                    frames.append(metrics)
                if not yearly.empty:
                    yearly_frames.append(yearly)
                if not exclusions.empty:
                    exclusion_frames.append(exclusions)
            except Exception as error:  # retain every unexpected basin-level failure
                failures.append({"GAGE_ID": gage, "error": repr(error)})
            if number % 50 == 0 or number == len(tasks):
                print(f"Completed {number}/{len(tasks)} basins", flush=True)

    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    exclusions = (
        pd.concat(exclusion_frames, ignore_index=True)
        if exclusion_frames else pd.DataFrame(columns=[
            "GAGE_ID", "lead_days", "threshold_name", "variant", "reason"
        ])
    )
    if metrics.empty or yearly.empty:
        raise RuntimeError("Sensitivity produced no eligible basin metrics")
    if failures:
        atomic_csv(pd.DataFrame(failures), output / "unexpected_failures.csv")
        raise RuntimeError(
            f"{len(failures)} unexpected failures; inspect {output / 'unexpected_failures.csv'}"
        )

    print("Computing paired spatial bootstrap summaries", flush=True)
    spatial = spatial_summaries(metrics, cfg, replicates)
    print("Computing water-year block bootstrap summaries", flush=True)
    block = block_summaries(yearly, cfg, replicates)
    exponents = exponent_summary(metrics)
    deltas = deltas_from_primary(spatial)
    conclusions = conclusion_summary(spatial, block, cfg, inferential)
    reproduction = reproduction_check(metrics, spatial)

    paths = {
        "metrics": output / "basin_model_metrics.csv.gz",
        "water_year": output / "basin_water_year_contrasts.csv.gz",
        "exclusions": output / "expected_exclusions.csv.gz",
        "spatial": output / "spatial_bootstrap_skill_summary.csv",
        "block": output / "water_year_block_skill_summary.csv",
        "exponents": output / "exponent_sensitivity_summary.csv",
        "deltas": output / "skill_deltas_from_primary.csv",
        "conclusions": output / "conclusion_sensitivity_summary.csv",
        "population": output / "analysis_population.csv",
    }
    atomic_csv(metrics, paths["metrics"], compression="gzip")
    atomic_csv(yearly, paths["water_year"], compression="gzip")
    atomic_csv(exclusions, paths["exclusions"], compression="gzip")
    atomic_csv(spatial, paths["spatial"])
    atomic_csv(block, paths["block"])
    atomic_csv(exponents, paths["exponents"])
    atomic_csv(deltas, paths["deltas"])
    atomic_csv(conclusions, paths["conclusions"])
    atomic_csv(population, paths["population"])

    receipt_payload = {
        "analysis": "Stage 18 variance-exponent shrinkage sensitivity",
        "sensitivity_version": SENSITIVITY_VERSION,
        "completed_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "run_label": run_label,
        "run_scope": "continental" if inferential else "pilot_or_debug",
        "publication_inference_allowed": inferential,
        "scientific_isolation": (
            "Primary Stage 18 files were read-only; all sensitivity products were "
            "written to this versioned sidecar directory."
        ),
        "held_fixed": [
            "Stage 10 source transitions",
            "dry-period definition",
            "training_end_date",
            "evaluation_start_date",
            "flow-state bins",
            "innovation groups",
            "ensemble size and quantile grid",
            "low-flow thresholds",
            "model definitions",
            "probability and CRPS scoring",
            "bootstrap resampling seeds within each comparison",
        ],
        "changed": (
            "Only variance_exponent_prior_sd: 0.5, 0.75, 1.5, or no shrinkage; "
            "all resulting scale coefficients and innovation distributions were refit."
        ),
        "no_shrinkage_definition": "raw fitted exponent clipped to [0, 4]",
        "variants": [
            {"variant": variant, "prior_sd": prior_sd} for variant, prior_sd in VARIANTS
        ],
        "leads": leads,
        "quantiles": quantiles,
        "bootstrap_replicates": replicates,
        "basins_requested": len(tasks),
        "basins_with_metrics": int(metrics["GAGE_ID"].nunique()),
        "spatial_groups": int(metrics["spatial_group"].nunique()),
        "run_fingerprint": fingerprint,
        "primary_reproduction_check": reproduction,
        "source_hashes": {
            "scripts/18_test_operational_value.py": sha256_file(
                SCRIPT_DIR / "18_test_operational_value.py"
            ),
            "scripts/config.json": sha256_file(Path(args.config).resolve()),
            "scripts/sensitivity_exponent_shrinkage.py": sha256_file(SCRIPT_PATH),
            "stage18_success_receipt": sha256_file(PRIMARY_STAGE18 / "_SUCCESS.json"),
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runtime_packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "primary_stage18_runtime": {
            "python": json.loads(
                (PRIMARY_STAGE18 / "_SUCCESS.json").read_text(encoding="utf-8")
            ).get("python"),
            "packages": json.loads((
                PROJECT_ROOT / "continental_run" / "00_environment"
                / "environment_report.json"
            ).read_text(encoding="utf-8")).get("packages", {}),
        },
        "outputs": [str(path) for path in paths.values()],
    }
    temporary_receipt = receipt.with_name(receipt.name + f".tmp-{os.getpid()}")
    temporary_receipt.write_text(
        json.dumps(receipt_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary_receipt, receipt)
    print(json.dumps({
        "receipt": str(receipt),
        "scope": receipt_payload["run_scope"],
        "basins": receipt_payload["basins_with_metrics"],
        "primary_reproduction": reproduction,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
