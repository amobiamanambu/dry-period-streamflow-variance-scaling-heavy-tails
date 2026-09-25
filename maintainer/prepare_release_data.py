#!/usr/bin/env python3
"""Build the compact public data layer from a completed analysis run.

This maintainer command is not part of the reader-facing reproduction. It
selects documented columns, preserves station identifiers as strings, removes
workstation paths, sanitizes receipts, and writes the release hash manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_GAGES = (
    "09394500", "12431000", "07328500", "05584500", "05129115",
    "02039500", "02092500", "02142000", "01381500",
)

GZIP = {"method": "gzip", "compresslevel": 9, "mtime": 0}


def normalize_gage(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(8)
    )


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen product is missing: {path}")
    return path


def write_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression=GZIP)


def build_example(analysis_root: Path) -> None:
    columns = [
        "basin_id", "GAGE_ID", "date", "q_cfs", "qualifier", "pr_mm", "pet_mm",
        "tmin_c", "tmax_c", "tmean_c", "q_mm_day", "snowmelt_risk", "q_norm",
        "q_next_1d", "dq_norm_1d", "valid_transition_1d",
        "q_next_2d", "dq_norm_2d", "valid_transition_2d",
        "q_next_3d", "dq_norm_3d", "valid_transition_3d",
        "dry_p0p1_a3_l1", "dry_p0p1_a3_l2", "dry_p0p1_a3_l3",
        "select_q_only_monotone_1d", "select_p_screened_monotone_1d",
        "select_p_screened_dry_state_1d", "select_q_only_monotone_2d",
        "select_p_screened_monotone_2d", "select_p_screened_dry_state_2d",
        "select_q_only_monotone_3d", "select_p_screened_monotone_3d",
        "select_p_screened_dry_state_3d",
    ]
    frames = []
    for gage in EXAMPLE_GAGES:
        source = require(analysis_root / "10_transitions" / "daily" / f"{gage}.csv.gz")
        frame = pd.read_csv(source, usecols=columns, dtype={"GAGE_ID": "string"})
        frame["GAGE_ID"] = normalize_gage(frame["GAGE_ID"])
        frames.append(frame)
    example = pd.concat(frames, ignore_index=True).sort_values(["GAGE_ID", "date"])
    write_gzip(example, REPO_ROOT / "data" / "example" / "example_transitions.csv.gz")

    result_source = require(
        analysis_root / "15_continental_results" / "continental_basin_results.csv"
    )
    metadata_columns = [
        "GAGE_ID", "STANAME", "STATE", "AGGECOREGION", "LAT_GAGE", "LNG_GAGE",
        "area_km2", "quality_tier", "is_reference", "recession_n_transitions",
        "recession_variance_exponent", "recession_variance_r2",
    ]
    metadata = pd.read_csv(
        result_source, usecols=metadata_columns, dtype={"GAGE_ID": "string"}, low_memory=False
    )
    metadata["GAGE_ID"] = normalize_gage(metadata["GAGE_ID"])
    metadata = metadata[metadata["GAGE_ID"].isin(EXAMPLE_GAGES)].sort_values("AGGECOREGION")
    if set(metadata["GAGE_ID"]) != set(EXAMPLE_GAGES):
        raise RuntimeError("Example-gage metadata is incomplete")
    metadata.to_csv(REPO_ROOT / "data" / "example" / "gage_metadata.csv", index=False)


def build_derived(
    analysis_root: Path,
    basin_inventory: Path,
    sensitivity_root: Path,
) -> None:
    destination = REPO_ROOT / "data" / "derived"
    destination.mkdir(parents=True, exist_ok=True)

    full_result = pd.read_csv(
        require(analysis_root / "15_continental_results" / "continental_basin_results.csv"),
        dtype={"GAGE_ID": "string", "HUC02": "string"}, low_memory=False,
    )
    full_result["GAGE_ID"] = normalize_gage(full_result["GAGE_ID"])
    scaling_columns = [
        "GAGE_ID", "basin_id", "quality_tier", "area_km2", "AGGECOREGION", "ecoregion",
        "STATE", "HUC02", "LAT_GAGE", "LNG_GAGE", "is_reference", "recession_n_transitions",
        "recession_variance_exponent", "recession_variance_coefficient",
        "recession_variance_r2", "recession_variance_bins",
        "recession_variance_exponent_ci_low", "recession_variance_exponent_ci_high",
        "recession_drift_exponent", "recession_drift_coefficient", "recession_drift_r2",
        "recession_variance_available", "analysis_population",
    ]
    scaling = full_result[scaling_columns].sort_values("GAGE_ID")
    write_gzip(scaling, destination / "basin_scaling.csv.gz")

    inventory = pd.read_csv(
        require(basin_inventory), dtype={"GAGE_ID": "string"}
    )
    inventory["GAGE_ID"] = normalize_gage(inventory["GAGE_ID"])
    status = scaling[[
        "GAGE_ID", "quality_tier", "AGGECOREGION", "STATE",
        "recession_variance_available",
    ]].copy()
    status["accepted"] = True
    status["estimable"] = status["recession_variance_available"].fillna(False).astype(bool)
    status = status.drop(columns="recession_variance_available")
    sample = inventory.rename(columns={"ecoregion": "inventory_ecoregion"}).merge(
        status, on="GAGE_ID", how="left", validate="one_to_one"
    )
    sample["accepted"] = sample["accepted"].fillna(False).astype(bool)
    sample["estimable"] = sample["estimable"].fillna(False).astype(bool)
    sample = sample.sort_values("GAGE_ID")
    write_gzip(sample, destination / "basin_sample.csv.gz")

    distribution_columns = [
        "GAGE_ID", "n_transitions", "n_standardized", "variance_exponent",
        "variance_coefficient", "variance_r2", "standardized_skew",
        "standardized_excess_kurtosis", "standardized_tail_fraction",
        "normal_tail_reference", "student_t_delta_aic_vs_normal",
        "median_pairwise_flow_group_ks", "maximum_pairwise_flow_group_ks",
        "maximum_abs_group_mean", "maximum_abs_group_sd_minus_one",
        "conservative_n", "conservative_standardized_excess_kurtosis",
        "conservative_standardized_tail_fraction", "quality_tier", "is_reference",
        "AGGECOREGION", "ecoregion", "aridity", "BFI_AVE", "log_area",
    ]
    distribution = pd.read_csv(
        require(analysis_root / "17_distribution_tests" / "distribution_basin_diagnostics.csv"),
        usecols=distribution_columns, dtype={"GAGE_ID": "string"}, low_memory=False,
    )
    distribution["GAGE_ID"] = normalize_gage(distribution["GAGE_ID"])
    write_gzip(distribution.sort_values("GAGE_ID"), destination / "basin_distribution.csv.gz")

    signed_columns = [
        "GAGE_ID", "screen_variant", "n_standardized", "positive_tail_count",
        "negative_tail_count", "positive_tail_fraction", "negative_tail_fraction",
        "two_sided_tail_fraction", "normal_one_sided_tail_reference",
        "positive_tail_multiple_of_normal", "negative_tail_multiple_of_normal",
        "two_sided_tail_multiple_of_normal", "abs_z_lag1_pearson", "z2_lag1_pearson",
        "quality_tier", "is_reference", "AGGECOREGION", "LAT_GAGE", "LNG_GAGE",
    ]
    signed = pd.read_csv(
        require(sensitivity_root / "tail_direction" / "basin_tail_direction_and_clustering.csv"),
        usecols=signed_columns, dtype={"GAGE_ID": "string"}, low_memory=False,
    )
    signed["GAGE_ID"] = normalize_gage(signed["GAGE_ID"])
    signed = signed[signed["screen_variant"].eq("primary")].sort_values("GAGE_ID")
    write_gzip(signed, destination / "basin_signed_tails.csv.gz")

    theory_columns = [
        "GAGE_ID", "n_all_sign_transitions", "m_all_recomputed", "theory_pair_available",
        "n_decline_transitions", "m_decline_recomputed", "b_recomputed", "two_b_recomputed",
        "delta_m_decline_minus_2b", "gamma_direct", "gamma_direct_r2",
        "weighted_negative_within_share", "weighted_positive_within_share",
        "weighted_between_sign_share", "m_negative_conditional",
        "m_positive_conditional", "water_year_bootstrap_sufficient",
        "gamma_direct_wy_ci_low", "gamma_direct_wy_ci_high", "quality_tier",
        "area_km2", "ecoregion", "LAT_GAGE", "LNG_GAGE", "STATE", "AGGECOREGION",
        "SNOW_PCT_PRECIP", "BFI_AVE",
    ]
    theory = pd.read_csv(
        require(sensitivity_root / "theory_bridge_v2" / "basin_theory_bridge.csv"),
        usecols=theory_columns, dtype={"GAGE_ID": "string"}, low_memory=False,
    )
    theory["GAGE_ID"] = normalize_gage(theory["GAGE_ID"])
    write_gzip(theory.sort_values("GAGE_ID"), destination / "basin_theory_bridge.csv.gz")

    prediction_columns = [
        "GAGE_ID", "spatial_group", "lead_days", "threshold_quantile", "threshold_name",
        "model", "n", "crps", "interval_coverage", "interval_mean_width", "median_mae",
    ]
    prediction = pd.read_csv(
        require(analysis_root / "18_lowflow_predictability_horizons" / "basin_predictability_metrics.csv.gz"),
        usecols=prediction_columns, dtype={"GAGE_ID": "string"}, low_memory=False,
    )
    prediction["GAGE_ID"] = normalize_gage(prediction["GAGE_ID"])
    models = {
        "regularized_power_gaussian",
        "regularized_power_state_gaussian",
        "regularized_power_state_empirical",
    }
    prediction = prediction[
        prediction["threshold_name"].eq("Q10") & prediction["model"].isin(models)
    ].sort_values(["lead_days", "GAGE_ID", "model"])
    write_gzip(prediction, destination / "basin_prediction_q10.csv.gz")

    huc = scaling[["GAGE_ID", "HUC02"]].copy()
    place = scaling[[
        "GAGE_ID", "HUC02", "STATE", "LAT_GAGE", "LNG_GAGE", "AGGECOREGION",
        "recession_variance_exponent",
    ]].rename(columns={"recession_variance_exponent": "variance_exponent"})
    place["lambda_star"] = 1.0 - place["variance_exponent"] / 2.0
    place = place.merge(
        signed[[
            "GAGE_ID", "positive_tail_multiple_of_normal",
            "negative_tail_multiple_of_normal",
        ]],
        on="GAGE_ID", how="left", validate="one_to_one",
    )
    matched_theory = theory[theory["m_decline_recomputed"].notna()][[
        "GAGE_ID", "gamma_direct", "weighted_negative_within_share",
    ]]
    place = place.merge(
        matched_theory, on="GAGE_ID", how="left", validate="one_to_one"
    )
    prediction_with_huc = prediction.merge(
        huc, on="GAGE_ID", how="left", validate="many_to_one"
    )
    lead_three = prediction_with_huc[prediction_with_huc["lead_days"].eq(3)]
    reference = lead_three[
        lead_three["model"].eq("regularized_power_state_gaussian")
    ][["GAGE_ID", "crps"]].rename(columns={"crps": "reference_crps"})
    empirical = lead_three[
        lead_three["model"].eq("regularized_power_state_empirical")
    ][["GAGE_ID", "crps"]].rename(columns={"crps": "empirical_crps"})
    gain = reference.merge(empirical, on="GAGE_ID", validate="one_to_one")
    gain["relative_crps_gain_percent"] = 100.0 * (
        gain["reference_crps"] - gain["empirical_crps"]
    ) / gain["reference_crps"]
    place = place.merge(
        gain[["GAGE_ID", "relative_crps_gain_percent"]],
        on="GAGE_ID", how="left", validate="one_to_one",
    )
    write_gzip(place.sort_values("GAGE_ID"), destination / "basin_place_metrics.csv.gz")

    small_sources = {
        "scaling_population_summary.csv": analysis_root / "16_scaling_tests" / "scaling_population_summary.csv",
        "scaling_meta_analysis.csv": analysis_root / "16_scaling_tests" / "scaling_random_effects_meta_analysis.csv",
        "scaling_lag_stability.csv": analysis_root / "16_scaling_tests" / "scaling_tau_stability.csv",
        "distribution_population_summary.csv": analysis_root / "17_distribution_tests" / "distribution_population_summary.csv",
        "prediction_skill_summary.csv": analysis_root / "18_lowflow_predictability_horizons" / "paired_predictability_skill_summary.csv",
        "prediction_water_year_summary.csv": analysis_root / "18_lowflow_predictability_horizons" / "water_year_block_skill_summary.csv",
        "theory_point_estimates.csv": sensitivity_root / "theory_bridge_v2" / "population_point_estimates.csv",
        "theory_bootstrap_intervals.csv": sensitivity_root / "theory_bridge_v2" / "population_bootstrap_intervals.csv",
        "robust_scale_population_summary.csv": sensitivity_root / "robust_scale" / "robust_scale_population_summary.csv",
        "signed_tail_population_summary.csv": sensitivity_root / "tail_direction" / "continental_summary.csv",
    }
    for name, source in small_sources.items():
        shutil.copyfile(require(source), destination / name)


def sanitize_value(value: Any, workspace: Path) -> Any:
    if isinstance(value, dict):
        return {
            sanitize_value(key, workspace): sanitize_value(item, workspace)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_value(item, workspace) for item in value]
    if not isinstance(value, str):
        return value
    text = value.replace(str(workspace), "${SOURCE_WORKSPACE}")
    for marker in ("continental_run/", "scripts/", "basin_inventory.csv"):
        position = text.find(marker)
        if position >= 0 and text.startswith("/"):
            return text[position:]
    if re.match(r"^/(Users|home)/[^/]+/", text):
        return f"external-input://{Path(text).name}"
    return text


def build_receipts(analysis_root: Path, source_root: Path) -> None:
    receipt_map = {
        "stage_08_matched_qc.json": analysis_root / "08_matched_qc" / "_SUCCESS.json",
        "stage_10_transitions.json": analysis_root / "10_transitions" / "_SUCCESS.json",
        "stage_14_recession.json": analysis_root / "14_recession" / "_SUCCESS.json",
        "stage_15_continental_results.json": analysis_root / "15_continental_results" / "_SUCCESS.json",
        "stage_16_scaling_tests.json": analysis_root / "16_scaling_tests" / "_SUCCESS.json",
        "stage_17_distribution_tests.json": analysis_root / "17_distribution_tests" / "_SUCCESS.json",
    }
    destination = REPO_ROOT / "provenance" / "receipts"
    destination.mkdir(parents=True, exist_ok=True)
    allowed_names = set(receipt_map)
    for stale in destination.glob("*.json"):
        if stale.name not in allowed_names:
            stale.unlink()
    for name, source in receipt_map.items():
        payload = json.loads(require(source).read_text(encoding="utf-8"))
        payload = sanitize_value(payload, source_root)
        (destination / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest() -> None:
    manifest_path = REPO_ROOT / "provenance" / "FILE_MANIFEST_SHA256.csv"
    rows = []
    excluded_roots = {
        ".git", ".venv", "venv", "outputs", "workspace", "__pycache__",
        ".pytest_cache", ".ruff_cache", "build", "dist",
    }
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(REPO_ROOT)
        if any(
            part in excluded_roots or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        rows.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=REPO_ROOT.parent,
        help="Source workspace used to resolve historical defaults and sanitize receipts.",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        required=True,
        help="Completed numbered-stage output root from a normalized full run.",
    )
    parser.add_argument(
        "--basin-inventory",
        type=Path,
        default=None,
        help="Source 9,067-gage inventory; defaults to <workspace>/basin_inventory.csv.",
    )
    parser.add_argument(
        "--sensitivity-root",
        type=Path,
        default=None,
        help=(
            "Directory containing tail_direction, theory_bridge_v2, and "
            "robust_scale analysis outputs. Defaults to "
            "<analysis-root>/sensitivity_analyses for a normalized rerun."
        ),
    )
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    analysis_root = args.analysis_root.expanduser().resolve()
    basin_inventory = (
        args.basin_inventory.expanduser().resolve()
        if args.basin_inventory is not None
        else workspace / "basin_inventory.csv"
    )
    sensitivity_root = (
        args.sensitivity_root.expanduser().resolve()
        if args.sensitivity_root is not None
        else analysis_root / "sensitivity_analyses"
    )
    build_example(analysis_root)
    build_derived(analysis_root, basin_inventory, sensitivity_root)
    build_receipts(analysis_root, workspace)
    build_manifest()
    print(f"Prepared compact release data in {REPO_ROOT}")


if __name__ == "__main__":
    main()
