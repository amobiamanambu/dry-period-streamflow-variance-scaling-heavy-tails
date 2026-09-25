#!/usr/bin/env python3
"""Stage 16: test the continental dry-state variance scaling and its limits."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from lib.common import (
    configure_logging, load_config, parse_args, prepare_stage, require_stage,
    stage_dir, write_receipt,
)
from lib.statistics import benjamini_hochberg, robust_ols, spearman_table


STAGE = 16


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def forcing_screen_summary(recession: pd.DataFrame) -> pd.DataFrame:
    primary = recession[recession["method"].eq("p_screened_dry_state")][
        ["GAGE_ID", "tau_days", "variance_exponent"]
    ].rename(columns={"variance_exponent": "primary_variance_exponent"})
    rows = []
    for (method, tau), group in recession.groupby(["method", "tau_days"], dropna=False):
        joined = group.merge(primary, on=["GAGE_ID", "tau_days"], how="left")
        pair = joined[["variance_exponent", "primary_variance_exponent"]].dropna()
        rows.append({
            "method": method,
            "tau_days": int(tau),
            "screening_pr_threshold_mm": group["screening_pr_threshold_mm"].iloc[0],
            "screening_antecedent_days": group["screening_antecedent_days"].iloc[0],
            "requires_monotone_decline": group["requires_monotone_decline"].iloc[0],
            "n_finite": int(group["variance_exponent"].notna().sum()),
            "median_variance_exponent": float(group["variance_exponent"].median()),
            "q25_variance_exponent": float(group["variance_exponent"].quantile(0.25)),
            "q75_variance_exponent": float(group["variance_exponent"].quantile(0.75)),
            "paired_n": len(pair),
            "spearman_vs_primary": float(pair.corr(method="spearman").iloc[0, 1])
            if len(pair) >= 20 else np.nan,
            "median_delta_vs_primary": float(
                (pair["variance_exponent"] - pair["primary_variance_exponent"]).median()
            ) if len(pair) else np.nan,
        })
    return pd.DataFrame(rows)


def tau_stability(primary: pd.DataFrame) -> pd.DataFrame:
    wide = primary.pivot(index="GAGE_ID", columns="tau_days", values="variance_exponent")
    rows = []
    for first, second in [(1, 2), (1, 3), (2, 3)]:
        if first not in wide or second not in wide:
            continue
        pair = wide[[first, second]].dropna()
        rows.append({
            "tau_first_days": first,
            "tau_second_days": second,
            "paired_n": len(pair),
            "spearman_rho": float(stats.spearmanr(pair[first], pair[second]).statistic)
            if len(pair) >= 20 else np.nan,
            "median_delta_second_minus_first": float((pair[second] - pair[first]).median())
            if len(pair) else np.nan,
            "median_absolute_delta": float((pair[second] - pair[first]).abs().median())
            if len(pair) else np.nan,
        })
    return pd.DataFrame(rows)


def random_effects_meta(frame: pd.DataFrame, target: float) -> dict:
    work = frame[[
        "variance_exponent", "variance_exponent_ci_low", "variance_exponent_ci_high"
    ]].apply(pd.to_numeric, errors="coerce").dropna()
    work["se"] = (
        work["variance_exponent_ci_high"] - work["variance_exponent_ci_low"]
    ) / (2 * 1.96)
    work = work[work["se"].gt(0)]
    if len(work) < 3:
        return {
            "meta_n": len(work), "fixed_mean": np.nan, "random_mean": np.nan,
            "random_ci_low": np.nan, "random_ci_high": np.nan,
            "prediction_low": np.nan, "prediction_high": np.nan,
            "tau2": np.nan, "i2_percent": np.nan, "target_exponent": target,
        }
    y = work["variance_exponent"].to_numpy(float)
    variance = work["se"].to_numpy(float) ** 2
    weight = 1.0 / variance
    fixed = float(np.sum(weight * y) / np.sum(weight))
    q = float(np.sum(weight * (y - fixed) ** 2))
    df = len(y) - 1
    denominator = float(np.sum(weight) - np.sum(weight**2) / np.sum(weight))
    tau2 = max(0.0, (q - df) / denominator) if denominator > 0 else 0.0
    random_weight = 1.0 / (variance + tau2)
    random_mean = float(np.sum(random_weight * y) / np.sum(random_weight))
    random_se = float(np.sqrt(1.0 / np.sum(random_weight)))
    prediction_se = float(np.sqrt(tau2 + random_se**2))
    return {
        "meta_n": len(work),
        "fixed_mean": fixed,
        "random_mean": random_mean,
        "random_ci_low": random_mean - 1.96 * random_se,
        "random_ci_high": random_mean + 1.96 * random_se,
        "prediction_low": random_mean - 1.96 * prediction_se,
        "prediction_high": random_mean + 1.96 * prediction_se,
        "tau2": tau2,
        "i2_percent": max(0.0, (q - df) / q * 100) if q > 0 else 0.0,
        "target_exponent": target,
    }


def population_masks(frame: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    reference = truthy(frame.get("is_reference", pd.Series(False, index=frame.index)))
    strict = truthy(frame.get(
        "passes_finite_time_stochastic_screens", pd.Series(False, index=frame.index)
    ))
    tier = frame.get("quality_tier", pd.Series("", index=frame.index)).astype(str)
    return {
        "accepted": pd.Series(True, index=frame.index),
        "tier1_high_quality": tier.eq(cfg["data_qc"]["primary_tier"]),
        "tier2_acceptable_quality": tier.eq("TIER2_ACCEPTABLE_QUALITY"),
        "reference": reference,
        "non_reference": ~reference,
        "finite_time_stochastic_screen_pass": strict,
    }


def population_scaling_summary(
    frame: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = float(cfg["scaling_test"]["target_exponent"])
    tolerance = float(cfg["scaling_test"]["near_target_tolerance"])
    rows, meta_rows = [], []
    for name, mask in population_masks(frame, cfg).items():
        group = frame.loc[mask].copy()
        beta = pd.to_numeric(group["variance_exponent"], errors="coerce")
        finite = group.loc[beta.notna()].copy()
        beta = pd.to_numeric(finite["variance_exponent"], errors="coerce")
        ci = finite[["variance_exponent_ci_low", "variance_exponent_ci_high"]].apply(
            pd.to_numeric, errors="coerce"
        )
        ci_valid = ci.notna().all(axis=1)
        low, high = ci.loc[ci_valid, "variance_exponent_ci_low"], ci.loc[
            ci_valid, "variance_exponent_ci_high"
        ]
        rows.append({
            "population": name,
            "n_total": len(group),
            "n_estimable": len(finite),
            "estimable_fraction": len(finite) / len(group) if len(group) else np.nan,
            "mean_exponent": float(beta.mean()),
            "median_exponent": float(beta.median()),
            "q25_exponent": float(beta.quantile(0.25)),
            "q75_exponent": float(beta.quantile(0.75)),
            "median_power_r2": float(pd.to_numeric(finite["variance_r2"], errors="coerce").median()),
            "median_relative_sd_exponent": float((beta / 2 - 1).median()),
            "point_within_target_tolerance_n": int((beta.sub(target).abs() <= tolerance).sum()),
            "point_within_target_tolerance_fraction": float(
                (beta.sub(target).abs() <= tolerance).mean()
            ) if len(beta) else np.nan,
            "bootstrap_ci_n": int(ci_valid.sum()),
            "ci_entirely_positive_n": int((low > 0).sum()),
            "ci_entirely_positive_fraction": float((low > 0).mean()) if len(low) else np.nan,
            "ci_below_target_n": int((high < target).sum()),
            "ci_includes_target_n": int(((low <= target) & (high >= target)).sum()),
            "ci_includes_target_fraction": float(
                ((low <= target) & (high >= target)).mean()
            ) if len(low) else np.nan,
            "ci_above_target_n": int((low > target).sum()),
        })
        meta_rows.append({"population": name, **random_effects_meta(finite, target)})
    return pd.DataFrame(rows), pd.DataFrame(meta_rows)


def diagnostic_summary(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for name, mask in population_masks(frame, cfg).items():
        group = frame.loc[mask]
        finite = group[pd.to_numeric(group["variance_exponent"], errors="coerce").notna()]
        rows.append({
            "population": name,
            "n_total": len(group),
            "n_estimable": len(finite),
            "markov_pass_n": int(truthy(finite["markov_pass"]).sum()),
            "markov_pass_fraction_estimable": float(truthy(finite["markov_pass"]).mean())
            if len(finite) else np.nan,
            "local_gaussian_pass_n": int(truthy(finite["km4_local_gaussian_pass"]).sum()),
            "local_gaussian_pass_fraction_estimable": float(
                truthy(finite["km4_local_gaussian_pass"]).mean()
            ) if len(finite) else np.nan,
            "finite_time_stochastic_screen_pass_n": int(
                truthy(finite["passes_finite_time_stochastic_screens"]).sum()
            ),
            "finite_time_stochastic_screen_pass_fraction_estimable": float(
                truthy(finite["passes_finite_time_stochastic_screens"]).mean()
            ) if len(finite) else np.nan,
        })
    return pd.DataFrame(rows)


def spatial_block_cv(frame: pd.DataFrame, response: str, predictors: list[str]) -> pd.DataFrame:
    group_column = "spatial_group"
    work = frame[[response, group_column, *predictors]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    rows = []
    for held_out in sorted(work[group_column].astype(str).unique()):
        train = work[work[group_column].astype(str).ne(held_out)]
        test = work[work[group_column].astype(str).eq(held_out)]
        if len(train) < 100 or len(test) < 20:
            continue
        mean, sd = train[predictors].mean(), train[predictors].std(ddof=0).replace(0, 1)
        model = LinearRegression().fit((train[predictors] - mean) / sd, train[response])
        prediction = model.predict((test[predictors] - mean) / sd)
        rows.append({
            "held_out_group": held_out, "n_train": len(train), "n_test": len(test),
            "rmse": float(np.sqrt(mean_squared_error(test[response], prediction))),
            "r2": float(r2_score(test[response], prediction)),
        })
    return pd.DataFrame(rows)


def write_decision(path, decision: dict) -> None:
    lines = [
        "# Stage 16 scaling decision", "",
        "This decision concerns conditional increment-variance scaling. The finite-time diagnostics are reported as screens and do not identify a formal diffusion coefficient.", "",
    ]
    for key, value in decision["checks"].items():
        lines.append(f"- **{key}**: {value}")
    lines.extend([
        "",
        f"**Near-multiplicative organizing tendency supported:** {decision['near_multiplicative_organizing_tendency_supported']}",
        f"**Universal exponent exactly equal to 2 supported:** {decision['universal_exact_beta2_supported']}",
        f"**Daily Gaussian diffusion interpretation supported:** {decision['daily_gaussian_diffusion_supported']}",
        f"**Proceed to distributional testing:** {decision['proceed_to_distributional_testing']}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 15)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)

    recession = pd.read_csv(
        stage_dir(cfg, 14) / "recession_basin_method_summary.csv",
        dtype={"GAGE_ID": str}, low_memory=False,
    )
    recession["GAGE_ID"] = recession["GAGE_ID"].str.zfill(8)
    results = pd.read_csv(
        stage_dir(cfg, 15) / "continental_basin_results.csv",
        dtype={"GAGE_ID": str}, low_memory=False,
    )
    results["GAGE_ID"] = results["GAGE_ID"].str.zfill(8)
    if args.limit:
        keep = set(results.head(args.limit)["GAGE_ID"])
        results = results[results["GAGE_ID"].isin(keep)]
        recession = recession[recession["GAGE_ID"].isin(keep)]

    primary_all_tau = recession[recession["method"].eq("p_screened_dry_state")].copy()
    primary = primary_all_tau[primary_all_tau["tau_days"].eq(1)].copy()
    attribute_columns = [
        "GAGE_ID", "quality_tier", "is_reference", "aridity", "BFI_AVE",
        "CONTACT", "STREAMS_KM_SQ_KM", "log_area", "snowfall_fraction_of_pr",
        "AGGECOREGION", "ecoregion",
    ]
    attribute_columns = [column for column in attribute_columns if column in results]
    analysis = primary.merge(
        results[attribute_columns], on="GAGE_ID", how="left", validate="one_to_one"
    )
    # Reconstruct the strict interpretation flag directly from the Stage-14
    # diagnostics so the scaling test does not depend on a renamed Stage-15
    # convenience column.
    analysis["passes_finite_time_stochastic_screens"] = (
        truthy(analysis["strict_diagnostics_complete"])
        & truthy(analysis["markov_pass"])
        & truthy(analysis["km4_local_gaussian_pass"])
    )
    analysis["spatial_group"] = analysis.get("AGGECOREGION", pd.Series(index=analysis.index)).fillna(
        analysis.get("ecoregion", "unknown")
    ).fillna("unknown")

    population, meta = population_scaling_summary(analysis, cfg)
    tau = tau_stability(primary_all_tau)
    forcing = forcing_screen_summary(recession)
    diagnostics = diagnostic_summary(analysis, cfg)

    predictors = [
        column for column in [
            "aridity", "BFI_AVE", "CONTACT", "STREAMS_KM_SQ_KM", "log_area",
            "snowfall_fraction_of_pr",
        ] if column in analysis
    ]
    drivers = spearman_table(analysis, ["variance_exponent"], predictors)
    if not drivers.empty:
        drivers["p_fdr_global"] = benjamini_hochberg(drivers["p_value"])
    ols, ols_summary = robust_ols(analysis, "variance_exponent", predictors)
    for key, value in ols_summary.items():
        ols[f"model_{key}"] = value
    cv = spatial_block_cv(analysis, "variance_exponent", predictors)

    paths = {
        "population": out / "scaling_population_summary.csv",
        "meta": out / "scaling_random_effects_meta_analysis.csv",
        "tau": out / "scaling_tau_stability.csv",
        "forcing": out / "forcing_screen_sensitivity_summary.csv",
        "diagnostics": out / "finite_time_diagnostic_summary.csv",
        "drivers": out / "secondary_driver_spearman.csv",
        "ols": out / "secondary_driver_robust_ols.csv",
        "cv": out / "secondary_driver_spatial_cv.csv",
        "decision_json": out / "SCALING_DECISION.json",
        "decision_md": out / "SCALING_DECISION.md",
    }
    population.to_csv(paths["population"], index=False)
    meta.to_csv(paths["meta"], index=False)
    tau.to_csv(paths["tau"], index=False)
    forcing.to_csv(paths["forcing"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    drivers.to_csv(paths["drivers"], index=False)
    ols.to_csv(paths["ols"], index=False)
    cv.to_csv(paths["cv"], index=False)

    population_index = population.set_index("population")
    accepted = population_index.loc["accepted"]
    tier1 = population_index.loc["tier1_high_quality"]
    tier2 = population_index.loc["tier2_acceptable_quality"]
    accepted_diag = diagnostics.set_index("population").loc["accepted"]
    threshold = cfg["scaling_test"]
    tau_1 = tau[tau["tau_first_days"].eq(1)]
    minimum_tau_rho = float(tau_1["spearman_rho"].min()) if len(tau_1) else np.nan
    checks = {
        "estimable_fraction": bool(
            accepted["estimable_fraction"] >= threshold["minimum_estimable_fraction"]
        ),
        "positive_bootstrap_ci_fraction": bool(
            accepted["ci_entirely_positive_fraction"] >= threshold["minimum_positive_ci_fraction"]
        ),
        "median_power_r2": bool(
            accepted["median_power_r2"] >= threshold["minimum_median_power_r2"]
        ),
        "cross_lag_rank_stability": bool(
            np.isfinite(minimum_tau_rho)
            and minimum_tau_rho >= threshold["minimum_cross_lag_spearman"]
        ),
        "continental_mean_near_two": bool(
            abs(accepted["mean_exponent"] - threshold["target_exponent"])
            <= threshold["near_target_tolerance"]
        ),
        "continental_median_near_two": bool(
            abs(accepted["median_exponent"] - threshold["target_exponent"])
            <= threshold["near_target_tolerance"]
        ),
        "tier1_replication": bool(
            tier1["estimable_fraction"] >= threshold["minimum_subgroup_estimable_fraction"]
            and abs(tier1["mean_exponent"] - threshold["target_exponent"])
            <= threshold["near_target_tolerance"]
            and abs(tier1["median_exponent"] - threshold["target_exponent"])
            <= threshold["near_target_tolerance"]
        ),
        "tier2_replication": bool(
            tier2["estimable_fraction"] >= threshold["minimum_subgroup_estimable_fraction"]
            and abs(tier2["mean_exponent"] - threshold["target_exponent"])
            <= threshold["near_target_tolerance"]
            and abs(tier2["median_exponent"] - threshold["target_exponent"])
            <= threshold["near_target_tolerance"]
        ),
    }
    universal = bool(
        accepted["ci_includes_target_fraction"]
        >= threshold["universal_ci_inclusion_fraction"]
    )
    finite_time_screen_majority = bool(
        accepted_diag["finite_time_stochastic_screen_pass_fraction_estimable"] >= 0.5
    )
    decision = {
        "analysis_label": "conditional dry-state increment-variance scaling",
        "checks": checks,
        "near_multiplicative_organizing_tendency_supported": bool(all(checks.values())),
        "universal_exact_beta2_supported": universal,
        "finite_time_stochastic_screen_majority_supported": finite_time_screen_majority,
        # Daily, finite-time transition data plus these diagnostics cannot by
        # themselves identify the infinitesimal Kramers--Moyal limit.
        "daily_gaussian_diffusion_supported": False,
        "proceed_to_distributional_testing": bool(all(checks.values())),
        "interpretation": (
            "Use conditional increment variance language. Do not call the fitted function "
            "a formal diffusion coefficient. The two reported diagnostics are finite-time screens."
        ),
    }
    paths["decision_json"].write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    write_decision(paths["decision_md"], decision)

    write_receipt(cfg, STAGE, paths.values(), {
        "basins": len(analysis),
        "estimable": int(analysis["variance_exponent"].notna().sum()),
        "near_multiplicative_supported": decision[
            "near_multiplicative_organizing_tendency_supported"
        ],
        "universal_exact_beta2_supported": universal,
        "finite_time_stochastic_screen_majority_supported": finite_time_screen_majority,
        "daily_gaussian_diffusion_supported": False,
    })
    logger.info(
        "Stage 16 complete: scaling=%s; universal beta=2=%s; daily Gaussian diffusion=%s",
        decision["near_multiplicative_organizing_tendency_supported"], universal, False,
    )


if __name__ == "__main__":
    main()
