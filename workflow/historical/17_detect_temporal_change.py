#!/usr/bin/env python3
"""Stage 17: test variance collapse, non-Gaussianity, and measurement robustness."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lib.common import (
    atomic_target, configure_logging, load_config, parse_args, prepare_stage,
    require_stage, stage_dir, write_receipt,
)
from lib.stochastic import conditional_kramers_moyal, loglog_power_fit


STAGE = 17
ANALYSIS_VERSION = 5


def exponent_fit(selected: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    setting = cfg["recession"]
    km = conditional_kramers_moyal(
        selected["q_norm"], selected["dq_norm_1d"], 1,
        setting["state_bins"], setting["minimum_bin_count"],
    )
    fit = loglog_power_fit(
        km["q_center"] if not km.empty else np.array([]),
        km["conditional_variance_rate"] if not km.empty else np.array([]),
        setting["minimum_valid_bins_for_fit"],
    )
    return km, fit


def interpolate_conditional_mean(q: np.ndarray, km: pd.DataFrame) -> np.ndarray:
    centers = km["q_center"].to_numpy(float)
    means = km["conditional_mean_increment"].to_numpy(float)
    order = np.argsort(centers)
    return np.interp(
        np.log(q), np.log(centers[order]), means[order],
        left=means[order][0], right=means[order][-1],
    )


def flow_group_diagnostics(q: np.ndarray, z: np.ndarray, groups: int) -> pd.DataFrame:
    assignments = pd.qcut(q, q=groups, labels=False, duplicates="drop")
    rows = []
    for group in sorted(pd.Series(assignments).dropna().unique()):
        selected = np.asarray(assignments == group)
        values = z[selected]
        rows.append({
            "flow_group": int(group), "n": len(values),
            "q_median": float(np.median(q[selected])),
            "z_mean": float(np.mean(values)), "z_sd": float(np.std(values, ddof=1)),
            "z_median": float(np.median(values)),
            "z_iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "z_tail_fraction_abs_gt_3": float(np.mean(np.abs(values) > 3)),
        })
    return pd.DataFrame(rows)


def pairwise_ks(groups: pd.Series, z: np.ndarray) -> tuple[float, float]:
    values = []
    unique = sorted(groups.dropna().unique())
    for index, first in enumerate(unique):
        for second in unique[index + 1:]:
            a = z[np.asarray(groups == first)]
            b = z[np.asarray(groups == second)]
            if min(len(a), len(b)) >= 20:
                values.append(float(stats.ks_2samp(a, b).statistic))
    return (
        float(np.median(values)) if values else np.nan,
        float(np.max(values)) if values else np.nan,
    )


def standardized_innovation_diagnostics(
    selected: pd.DataFrame, cfg: dict
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Fit the conditional variance law and diagnose standardized innovations."""
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
    q, raw_z = q[finite], raw_z[finite]
    raw_mean = float(np.mean(raw_z))
    raw_sd = float(np.std(raw_z, ddof=1))
    if not np.isfinite(raw_sd) or raw_sd <= 0:
        raise ValueError("Power-scaled residual variance unavailable")
    z = (raw_z - raw_mean) / raw_sd
    retained["standardized_innovation"] = z

    excess = float(stats.kurtosis(z, fisher=True, bias=False)) if len(z) > 3 else np.nan
    skew = float(stats.skew(z, bias=False)) if len(z) > 2 else np.nan
    student_df = (
        float(np.clip(4.0 + 6.0 / excess, 4.1, 100.0))
        if np.isfinite(excess) and excess > 0 else 100.0
    )
    student_scale = np.sqrt((student_df - 2.0) / student_df)
    normal_ll = float(np.sum(stats.norm.logpdf(z)))
    student_ll = float(np.sum(stats.t.logpdf(z, df=student_df, scale=student_scale)))
    normal_aic = -2.0 * normal_ll
    student_aic = 2.0 - 2.0 * student_ll

    flow_groups = int(cfg["distribution_test"]["flow_groups"])
    labels = pd.qcut(q, q=flow_groups, labels=False, duplicates="drop")
    groups = flow_group_diagnostics(q, z, flow_groups)
    median_ks, maximum_ks = pairwise_ks(pd.Series(labels), z)
    threshold = float(cfg["distribution_test"]["normal_tail_z"])
    metrics = {
        "n_standardized": len(z),
        "variance_exponent": fit["exponent"],
        "variance_coefficient": fit["coefficient"],
        "variance_r2": fit["r2"],
        "power_scaled_mean_before_global_centering": raw_mean,
        "power_scaled_sd_before_global_rescaling": raw_sd,
        "standardized_mean": float(np.mean(z)),
        "standardized_sd": float(np.std(z, ddof=1)),
        "standardized_skew": skew,
        "standardized_excess_kurtosis": excess,
        "standardized_tail_fraction": float(np.mean(np.abs(z) > threshold)),
        "normal_tail_reference": float(2 * stats.norm.sf(threshold)),
        "normal_ks_statistic": float(stats.kstest(z, "norm").statistic),
        "student_t_df_moment": student_df,
        "normal_aic": normal_aic,
        "student_t_aic": student_aic,
        "student_t_delta_aic_vs_normal": student_aic - normal_aic,
        "median_pairwise_flow_group_ks": median_ks,
        "maximum_pairwise_flow_group_ks": maximum_ks,
        "maximum_abs_group_mean": float(groups["z_mean"].abs().max()),
        "maximum_abs_group_sd_minus_one": float((groups["z_sd"] - 1).abs().max()),
    }
    return metrics, groups, retained


def distribution_worker(task):
    gage, source, summary_file, groups_file, cfg, force = task
    summary_path, groups_path = Path(summary_file), Path(groups_file)
    if summary_path.exists() and groups_path.exists() and not force:
        cached = pd.read_csv(summary_path)
        if (
            len(cached) and "analysis_version" in cached
            and int(cached["analysis_version"].iloc[0]) >= ANALYSIS_VERSION
        ):
            return cached, pd.read_csv(groups_path)

    frame = pd.read_csv(source, parse_dates=["date"], low_memory=False).sort_values("date")
    # A one-day increment uses discharge at both t and t+1.  Removing only an
    # estimated origin would leave transitions whose destination was estimated,
    # so retain both endpoint qualifiers for the measurement-robustness test.
    frame["qualifier_next_1d"] = frame["qualifier"].shift(-1)
    frame["q_cfs_next_1d"] = pd.to_numeric(frame["q_cfs"], errors="coerce").shift(-1)
    flag = frame["select_p_screened_dry_state_1d"].fillna(False).astype(bool)
    selected = frame.loc[flag, [
        "date", "q_norm", "dq_norm_1d", "q_cfs", "q_cfs_next_1d",
        "qualifier", "qualifier_next_1d",
    ]].replace([np.inf, -np.inf], np.nan).dropna(subset=["q_norm", "dq_norm_1d"])
    selected = selected[selected["q_norm"].gt(0)]
    setting = cfg["distribution_test"]
    minimum = int(setting["minimum_transitions"])
    if len(selected) < minimum:
        raise ValueError(f"Only {len(selected)} primary dry-state transitions; need {minimum}")
    primary, groups, retained = standardized_innovation_diagnostics(selected, cfg)

    origin_estimated = retained["qualifier"].fillna("").astype(str).str.contains(
        ":e", case=False, regex=False
    )
    destination_estimated = retained["qualifier_next_1d"].fillna("").astype(str).str.contains(
        ":e", case=False, regex=False
    )
    nonestimated_mask = ~(origin_estimated | destination_estimated)
    nonestimated = retained.loc[nonestimated_mask].drop(columns="standardized_innovation")
    sensitivity_minimum = int(setting["tail_sensitivity_minimum_transitions"])
    nonestimated_metrics = None
    if len(nonestimated) >= sensitivity_minimum:
        try:
            nonestimated_metrics, _, nonestimated = standardized_innovation_diagnostics(
                nonestimated, cfg
            )
        except ValueError:
            nonestimated_metrics = None

    endpoints = pd.concat([
        pd.to_numeric(nonestimated.get("q_cfs"), errors="coerce"),
        pd.to_numeric(nonestimated.get("q_cfs_next_1d"), errors="coerce"),
    ], ignore_index=True)
    positive_endpoints = endpoints[endpoints.gt(0) & np.isfinite(endpoints)]
    cutoff = (
        float(positive_endpoints.quantile(float(setting["tail_sensitivity_low_flow_quantile"])))
        if len(positive_endpoints) else np.nan
    )
    conservative = nonestimated.copy()
    if np.isfinite(cutoff):
        conservative = conservative[
            pd.to_numeric(conservative["q_cfs"], errors="coerce").gt(cutoff)
            & pd.to_numeric(conservative["q_cfs_next_1d"], errors="coerce").gt(cutoff)
        ]
    if bool(setting["tail_sensitivity_exclude_zero_increments"]):
        conservative = conservative[
            ~np.isclose(conservative["dq_norm_1d"].to_numpy(float), 0.0, rtol=0.0, atol=0.0)
        ]
    conservative_metrics = None
    if len(conservative) >= sensitivity_minimum:
        try:
            conservative_metrics, _, conservative = standardized_innovation_diagnostics(
                conservative, cfg
            )
        except ValueError:
            conservative_metrics = None

    def sensitivity_value(metrics: dict | None, key: str) -> float:
        return float(metrics[key]) if metrics is not None else np.nan

    summary = pd.DataFrame([{
        "analysis_version": ANALYSIS_VERSION,
        "GAGE_ID": str(gage).zfill(8),
        "n_transitions": len(selected),
        **primary,
        "nonestimated_n": len(nonestimated),
        "nonestimated_fraction": len(nonestimated) / len(selected),
        "nonestimated_sensitivity_available": nonestimated_metrics is not None,
        "nonestimated_variance_exponent": sensitivity_value(nonestimated_metrics, "variance_exponent"),
        "nonestimated_variance_r2": sensitivity_value(nonestimated_metrics, "variance_r2"),
        "nonestimated_standardized_excess_kurtosis": sensitivity_value(
            nonestimated_metrics, "standardized_excess_kurtosis"
        ),
        "nonestimated_standardized_tail_fraction": sensitivity_value(
            nonestimated_metrics, "standardized_tail_fraction"
        ),
        "nonestimated_student_t_delta_aic_vs_normal": sensitivity_value(
            nonestimated_metrics, "student_t_delta_aic_vs_normal"
        ),
        "conservative_low_flow_cutoff_cfs": cutoff,
        "conservative_n": len(conservative),
        "conservative_fraction": len(conservative) / len(selected),
        "conservative_sensitivity_available": conservative_metrics is not None,
        "conservative_variance_exponent": sensitivity_value(conservative_metrics, "variance_exponent"),
        "conservative_variance_r2": sensitivity_value(conservative_metrics, "variance_r2"),
        "conservative_standardized_excess_kurtosis": sensitivity_value(
            conservative_metrics, "standardized_excess_kurtosis"
        ),
        "conservative_standardized_tail_fraction": sensitivity_value(
            conservative_metrics, "standardized_tail_fraction"
        ),
        "conservative_student_t_delta_aic_vs_normal": sensitivity_value(
            conservative_metrics, "student_t_delta_aic_vs_normal"
        ),
    }])
    groups.insert(0, "GAGE_ID", str(gage).zfill(8))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    groups_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_target(summary_path) as temporary:
        summary.to_csv(temporary, index=False)
    with atomic_target(groups_path) as temporary:
        groups.to_csv(temporary, index=False)
    return summary, groups


def write_decision(path: Path, decision: dict) -> None:
    lines = [
        "# Stage 17 distribution decision", "",
        "The fitted power law is evaluated as a conditional location-scale relationship, not automatically as a Gaussian diffusion.", "",
    ]
    for key, value in decision["checks"].items():
        lines.append(f"- **{key}**: {value}")
    lines.extend([
        "",
        f"**State scaling survives removal of estimated values:** {decision['measurement_robust_scaling_supported']}",
        f"**Standardized-residual collapse supported:** {decision['standardized_residual_collapse_supported']}",
        f"**Non-Gaussian innovations supported:** {decision['non_gaussian_innovations_supported']}",
        f"**Heavy tails survive measurement/rounding sensitivity tests:** {decision['non_gaussian_measurement_and_rounding_robust_supported']}",
        f"**Proceed to low-flow predictability-horizon test:** {decision['proceed_to_lowflow_predictability_test']}",
        "",
        "Failure of a single basin-wide residual collapse is not used as a stop rule. "
        "Stage 18 responds to that result by estimating empirical innovations separately across flow-state groups.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 16)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)

    inventory = pd.read_csv(
        stage_dir(cfg, 10) / "transition_inventory.csv", dtype={"GAGE_ID": str}
    )
    attributes = pd.read_csv(
        stage_dir(cfg, 15) / "continental_basin_results.csv",
        dtype={"GAGE_ID": str}, low_memory=False,
    )
    inventory["GAGE_ID"] = inventory["GAGE_ID"].str.zfill(8)
    attributes["GAGE_ID"] = attributes["GAGE_ID"].str.zfill(8)
    inventory = inventory[inventory["GAGE_ID"].isin(set(attributes["GAGE_ID"]))]
    if args.limit:
        inventory = inventory.head(args.limit)

    summary_dir, group_dir = out / "by_basin_summary", out / "by_basin_flow_groups"
    tasks = [(
        row.GAGE_ID, row.file,
        str(summary_dir / f"{row.GAGE_ID}.csv"),
        str(group_dir / f"{row.GAGE_ID}.csv"), cfg, args.force,
    ) for row in inventory.itertuples(index=False)]
    summaries, groups, failures = [], [], []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(distribution_worker, task): task[0] for task in tasks}
        for number, future in enumerate(as_completed(futures), start=1):
            gage = futures[future]
            try:
                summary, group = future.result()
                summaries.append(summary)
                groups.append(group)
            except ValueError as error:
                failures.append({"GAGE_ID": gage, "status": "excluded_insufficient_or_unfittable", "reason": str(error)})
            except Exception as error:
                failures.append({"GAGE_ID": gage, "status": "failed_unexpected", "reason": repr(error)})
                logger.error("Distribution analysis failed for %s: %r", gage, error)
            if number % 100 == 0:
                logger.info("Completed %d/%d basin distribution diagnostics", number, len(tasks))

    basin = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    flow = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()
    if basin.empty:
        raise RuntimeError("No basin supported Stage-17 distribution diagnostics")
    keep_columns = [
        column for column in [
            "GAGE_ID", "quality_tier", "is_reference", "AGGECOREGION", "ecoregion",
            "aridity", "BFI_AVE", "log_area",
        ] if column in attributes
    ]
    basin = basin.merge(attributes[keep_columns], on="GAGE_ID", how="left", validate="one_to_one")

    pair = basin[["variance_exponent", "nonestimated_variance_exponent"]].dropna()
    exponent_rho = float(pair.corr(method="spearman").iloc[0, 1]) if len(pair) >= 20 else np.nan
    median_delta = float(
        (pair["nonestimated_variance_exponent"] - pair["variance_exponent"]).median()
    ) if len(pair) else np.nan
    t_win = basin["student_t_delta_aic_vs_normal"].lt(-2)
    population = pd.DataFrame([{
        "basins_requested": len(tasks), "basins_analyzed": len(basin),
        "median_variance_exponent": float(basin["variance_exponent"].median()),
        "median_nonestimated_variance_exponent": float(
            basin["nonestimated_variance_exponent"].median()
        ),
        "paired_nonestimated_n": len(pair),
        "paired_nonestimated_fraction": len(pair) / len(basin),
        "median_nonestimated_exponent_delta": median_delta,
        "nonestimated_exponent_spearman": exponent_rho,
        "median_standardized_excess_kurtosis": float(
            basin["standardized_excess_kurtosis"].median()
        ),
        "median_standardized_tail_fraction": float(
            basin["standardized_tail_fraction"].median()
        ),
        "student_t_aic_win_n": int(t_win.sum()),
        "student_t_aic_win_fraction": float(t_win.mean()),
        "nonestimated_tail_sensitivity_n": int(
            basin["nonestimated_sensitivity_available"].fillna(False).sum()
        ),
        "nonestimated_tail_sensitivity_fraction": float(
            basin["nonestimated_sensitivity_available"].fillna(False).mean()
        ),
        "median_nonestimated_standardized_excess_kurtosis": float(
            basin["nonestimated_standardized_excess_kurtosis"].median()
        ),
        "median_nonestimated_standardized_tail_fraction": float(
            basin["nonestimated_standardized_tail_fraction"].median()
        ),
        "nonestimated_student_t_aic_win_fraction": float(
            basin.loc[
                basin["nonestimated_sensitivity_available"].fillna(False),
                "nonestimated_student_t_delta_aic_vs_normal",
            ].lt(-2).mean()
        ),
        "conservative_tail_sensitivity_n": int(
            basin["conservative_sensitivity_available"].fillna(False).sum()
        ),
        "conservative_tail_sensitivity_fraction": float(
            basin["conservative_sensitivity_available"].fillna(False).mean()
        ),
        "median_conservative_standardized_excess_kurtosis": float(
            basin["conservative_standardized_excess_kurtosis"].median()
        ),
        "median_conservative_standardized_tail_fraction": float(
            basin["conservative_standardized_tail_fraction"].median()
        ),
        "conservative_student_t_aic_win_fraction": float(
            basin.loc[
                basin["conservative_sensitivity_available"].fillna(False),
                "conservative_student_t_delta_aic_vs_normal",
            ].lt(-2).mean()
        ),
        "median_pairwise_flow_group_ks": float(
            basin["median_pairwise_flow_group_ks"].median()
        ),
        "median_maximum_flow_group_ks": float(
            basin["maximum_pairwise_flow_group_ks"].median()
        ),
        "median_maximum_abs_group_mean": float(
            basin["maximum_abs_group_mean"].median()
        ),
        "median_maximum_abs_group_sd_minus_one": float(
            basin["maximum_abs_group_sd_minus_one"].median()
        ),
    }])

    setting = cfg["distribution_test"]
    measurement = bool(
        np.isfinite(median_delta) and abs(median_delta)
        <= setting["maximum_median_exponent_change_without_estimated_values"]
        and np.isfinite(exponent_rho)
        and exponent_rho >= setting["minimum_exponent_rank_correlation_without_estimated_values"]
        and population["paired_nonestimated_fraction"].iloc[0]
        >= setting["minimum_paired_nonestimated_fraction"]
    )
    collapse = bool(
        population["median_pairwise_flow_group_ks"].iloc[0]
        <= setting["maximum_median_pairwise_ks_for_collapse"]
        and population["median_maximum_abs_group_mean"].iloc[0]
        <= setting["maximum_median_abs_group_mean_for_collapse"]
        and population["median_maximum_abs_group_sd_minus_one"].iloc[0]
        <= setting["maximum_median_abs_group_sd_minus_one_for_collapse"]
    )
    non_gaussian = bool(
        population["student_t_aic_win_fraction"].iloc[0]
        >= setting["minimum_student_t_aic_win_fraction"]
    )
    gaussian_tail = float(2 * stats.norm.sf(float(setting["normal_tail_z"])))
    tail_multiple = float(setting["minimum_tail_frequency_multiple_of_gaussian"])
    sensitivity_fraction = float(setting["minimum_tail_sensitivity_basin_fraction"])
    nonestimated_tail_robust = bool(
        population["nonestimated_tail_sensitivity_fraction"].iloc[0] >= sensitivity_fraction
        and population["nonestimated_student_t_aic_win_fraction"].iloc[0]
        >= setting["minimum_student_t_aic_win_fraction"]
        and population["median_nonestimated_standardized_tail_fraction"].iloc[0]
        >= gaussian_tail * tail_multiple
    )
    conservative_tail_robust = bool(
        population["conservative_tail_sensitivity_fraction"].iloc[0] >= sensitivity_fraction
        and population["conservative_student_t_aic_win_fraction"].iloc[0]
        >= setting["minimum_student_t_aic_win_fraction"]
        and population["median_conservative_standardized_tail_fraction"].iloc[0]
        >= gaussian_tail * tail_multiple
    )
    robust_non_gaussian = bool(non_gaussian and nonestimated_tail_robust and conservative_tail_robust)
    scaling_decision = json.loads(
        (stage_dir(cfg, 16) / "SCALING_DECISION.json").read_text(encoding="utf-8")
    )
    decision = {
        "checks": {
            "estimated_value_endpoint_exclusion_coverage": bool(
                population["paired_nonestimated_fraction"].iloc[0]
                >= setting["minimum_paired_nonestimated_fraction"]
            ),
            "estimated_value_exponent_magnitude_and_rank_robustness": measurement,
            "flow_group_distribution_collapse": collapse,
            "student_t_beats_normal_in_required_fraction": non_gaussian,
            "heavy_tails_survive_estimated_endpoint_removal": nonestimated_tail_robust,
            "heavy_tails_survive_estimated_small_flow_and_zero_increment_removal": (
                conservative_tail_robust
            ),
        },
        "measurement_robust_scaling_supported": measurement,
        "standardized_residual_collapse_supported": collapse,
        "non_gaussian_innovations_supported": non_gaussian,
        "non_gaussian_measurement_and_rounding_robust_supported": robust_non_gaussian,
        "proceed_to_probabilistic_lowflow_test": bool(
            scaling_decision["proceed_to_distributional_testing"] and measurement
        ),
        "proceed_to_lowflow_predictability_test": bool(
            scaling_decision["proceed_to_distributional_testing"] and measurement
        ),
        "state_specific_innovations_required": bool(not collapse),
    }

    paths = {
        "basin": out / "distribution_basin_diagnostics.csv",
        "flow": out / "distribution_flow_group_diagnostics.csv.gz",
        "population": out / "distribution_population_summary.csv",
        "failures": out / "distribution_exclusions_and_failures.csv",
        "decision_json": out / "DISTRIBUTION_DECISION.json",
        "decision_md": out / "DISTRIBUTION_DECISION.md",
    }
    basin.to_csv(paths["basin"], index=False)
    flow.to_csv(paths["flow"], index=False, compression="gzip")
    population.to_csv(paths["population"], index=False)
    pd.DataFrame(failures, columns=["GAGE_ID", "status", "reason"]).to_csv(
        paths["failures"], index=False
    )
    paths["decision_json"].write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    write_decision(paths["decision_md"], decision)
    unexpected = sum(item["status"] == "failed_unexpected" for item in failures)
    if unexpected:
        raise RuntimeError(
            f"{unexpected} unexpected Stage-17 failures; inspect {paths['failures']}"
        )
    write_receipt(cfg, STAGE, [*paths.values(), summary_dir, group_dir], {
        "basins_requested": len(tasks), "basins_analyzed": len(basin),
        "excluded": len(failures), "non_gaussian_supported": non_gaussian,
        "measurement_robust": measurement, "distribution_collapse_supported": collapse,
    })
    logger.info(
        "Stage 17 complete: measurement robust=%s; collapse=%s; non-Gaussian=%s",
        measurement, collapse, non_gaussian,
    )


if __name__ == "__main__":
    main()
