#!/usr/bin/env python3
"""Stage 15: join basin signatures, diagnostics, forcing summaries, QC, and GAGES-II attributes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lib.common import (
    load_config, parse_args, prepare_stage, configure_logging, require_stage,
    stage_dir, write_receipt,
)


STAGE = 15


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 14)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    attributes = pd.read_csv(stage_dir(cfg, 6) / "basin_index.csv", dtype={"GAGE_ID": str}, low_memory=False)
    qc = pd.read_csv(stage_dir(cfg, 8) / "matched_data_qc.csv", dtype={"GAGE_ID": str})
    snow = pd.read_csv(stage_dir(cfg, 9) / "snow_proxy_summary.csv", dtype={"GAGE_ID": str})
    irreversible = pd.read_csv(stage_dir(cfg, 13) / "irreversibility_basin_summary.csv", dtype={"GAGE_ID": str})
    irreversibility_status = pd.read_csv(
        stage_dir(cfg, 13) / "irreversibility_basin_status.csv", dtype={"GAGE_ID": str}
    )
    recession = pd.read_csv(stage_dir(cfg, 14) / "recession_basin_method_summary.csv", dtype={"GAGE_ID": str})
    primary = recession[(recession["method"] == "p_screened_dry_state") & (recession["tau_days"] == 1)].copy()
    primary = primary.rename(columns={column: f"recession_{column}" for column in primary.columns if column != "GAGE_ID"})
    # The primary dry-state scaling analysis is not conditioned on successful
    # irreversibility estimation.  Begin with every Stage-14 basin and attach
    # irreversibility as an optional, independent descriptor.
    results = primary.merge(snow, on="GAGE_ID", how="left", validate="one_to_one")
    results = results.merge(qc, on="GAGE_ID", how="left", suffixes=("", "_qc"), validate="one_to_one")
    results = results.merge(attributes, on="GAGE_ID", how="left", suffixes=("", "_attr"), validate="one_to_one")
    results = results.merge(irreversible, on="GAGE_ID", how="left", validate="one_to_one")
    results = results.merge(
        irreversibility_status.rename(columns={
            "status": "irreversibility_status",
            "reason": "irreversibility_exclusion_reason",
        }),
        on="GAGE_ID", how="left", validate="one_to_one",
    )
    results["recession_diagnostic_complete"] = (
        truthy(results["recession_strict_diagnostics_complete"])
    )
    results["passes_finite_time_stochastic_screens"] = (
        results["recession_diagnostic_complete"]
        & truthy(results["recession_markov_pass"])
        & truthy(results["recession_km4_local_gaussian_pass"])
    )
    # Backward-compatible field name. Passing these two finite-time screens is
    # necessary but not sufficient evidence for a formal diffusion process.
    results["passes_strict_daily_diffusion_diagnostics"] = results[
        "passes_finite_time_stochastic_screens"
    ]
    results["recession_variance_available"] = pd.to_numeric(
        results["recession_variance_exponent"], errors="coerce"
    ).notna()
    coefficient = pd.to_numeric(
        results["recession_variance_coefficient"], errors="coerce"
    )
    results["recession_log_variance_coefficient"] = np.log(
        coefficient.where(coefficient > 0)
    )
    if "peak_lag_right_censored" not in results:
        maximum_lag = max(cfg["transition_definitions"]["lags_days"])
        results["peak_lag_right_censored"] = pd.to_numeric(
            results["peak_lag_days"], errors="coerce"
        ).eq(maximum_lag)
    results["analysis_population"] = "accepted_Q_tier"
    results.loc[results["quality_tier"].eq(cfg["data_qc"]["primary_tier"]), "analysis_population"] = "primary_Q_tier"
    output = out / "continental_basin_results.csv"
    results.to_csv(output, index=False)
    core_columns = [
        "GAGE_ID", "basin_id", "quality_tier", "CLASS", "AGGECOREGION", "area_km2",
        "BFI_AVE", "aridity", "snow_fraction", "snowfall_fraction_of_pr",
        "bias_corrected_1d", "peak_lag_days", "peak_bias_corrected", "spectrum_auc_loglag",
        "ordinal_kl_nats", "peak_lag_right_censored", "recession_variance_exponent",
        "low_anomaly_pair_fraction_1d", "high_anomaly_pair_fraction_1d",
        "net_upward_probability_current_1d", "iaaft_p_resolution",
        "recession_variance_exponent_ci_low", "recession_variance_exponent_ci_high",
        "recession_variance_coefficient", "recession_log_variance_coefficient",
        "recession_variance_r2", "recession_ck_error",
        "recession_median_km4_finite_time_ratio", "recession_median_km4_gaussian_deviation",
        "recession_variance_available", "recession_diagnostic_complete",
        "passes_finite_time_stochastic_screens",
        "passes_strict_daily_diffusion_diagnostics",
    ]
    core_columns = [column for column in core_columns if column in results]
    core = out / "continental_core_signatures.csv"
    results[core_columns].to_csv(core, index=False)
    metrics = {
        "basins": len(results),
        "irreversibility_excluded": int(
            (~pd.read_csv(
                stage_dir(cfg, 13) / "irreversibility_basin_status.csv"
            )["status"].eq("eligible")).sum()
        ),
        "variance_exponent_available": int(results["recession_variance_available"].sum()),
        "strict_diagnostic_complete": int(results["recession_diagnostic_complete"].sum()),
        "finite_time_stochastic_screen_pass": int(
            results["passes_finite_time_stochastic_screens"].sum()
        ),
        "peak_lag_right_censored": int(results["peak_lag_right_censored"].sum()),
        "primary_q_tier": int((results["analysis_population"] == "primary_Q_tier").sum()),
    }
    write_receipt(cfg, STAGE, [output, core], metrics)
    logger.info("Stage 15 complete: consolidated %s basins", f"{len(results):,}")


if __name__ == "__main__":
    main()
