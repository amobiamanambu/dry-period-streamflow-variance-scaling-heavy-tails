#!/usr/bin/env python3
"""Stage 14: estimate forcing-screened drift and conditional-variance scaling."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from lib.analysis import analyze_recession
from lib.common import (
    atomic_target, load_config, parse_args, prepare_stage, configure_logging,
    require_stage, stage_dir, write_receipt,
)


STAGE = 14


def worker(task):
    gage, source_file, summary_file, bins_file, cfg, force = task
    summary_path, bins_path = Path(summary_file), Path(bins_file)
    if summary_path.exists() and bins_path.exists() and not force:
        cached = pd.read_csv(summary_path)
        required = {
            "variance_exponent", "strict_diagnostics_complete",
            "passes_finite_time_stochastic_screens",
            "screening_pr_threshold_mm", "screening_antecedent_days",
        }
        if required.issubset(cached.columns):
            return gage, cached, None
    frame = pd.read_csv(source_file, parse_dates=["date"], low_memory=False)
    rng = np.random.default_rng(cfg["project"]["random_seed"] + 10_000_000 + int(gage))
    summary, bins, diagnostic = analyze_recession(
        frame, cfg, rng, cfg["recession"]["bootstrap_replicates_continental"], True,
        include_forcing_sensitivity=True,
        store_sensitivity_bins=False,
    )
    summary.insert(0, "GAGE_ID", gage)
    summary["markov_warning"] = diagnostic["markov_warning"]
    summary["km4_warning"] = diagnostic["km4_warning"]
    for column in (
        "markov_diagnostic_available", "km4_diagnostic_available",
        "strict_diagnostics_complete", "markov_pass", "km4_local_gaussian_pass",
        "passes_finite_time_stochastic_screens",
        "passes_strict_daily_diffusion_diagnostics",
    ):
        summary[column] = diagnostic[column]
    bins.insert(0, "GAGE_ID", gage)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    bins_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_target(summary_path) as temporary:
        summary.to_csv(temporary, index=False)
    with atomic_target(bins_path) as temporary:
        bins.to_csv(temporary, index=False)
    return gage, summary, None


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 13)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    inventory = pd.read_csv(stage_dir(cfg, 10) / "transition_inventory.csv", dtype={"GAGE_ID": str})
    if args.limit:
        inventory = inventory.head(args.limit)
    summary_dir, bins_dir = out / "by_basin_summary", out / "by_basin_bins"
    tasks = [
        (str(row.GAGE_ID).zfill(8), row.file,
         str(summary_dir / f"{str(row.GAGE_ID).zfill(8)}.csv"),
         str(bins_dir / f"{str(row.GAGE_ID).zfill(8)}.csv"), cfg, args.force)
        for row in inventory.itertuples(index=False)
    ]
    frames, failures = [], []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(worker, task): task[0] for task in tasks}
        for number, future in enumerate(as_completed(futures), start=1):
            gage = futures[future]
            try:
                _, frame, _ = future.result()
                frames.append(frame)
            except Exception as error:
                failures.append({"GAGE_ID": gage, "error": repr(error)})
                logger.error("Recession analysis failed for %s: %r", gage, error)
            if number % 100 == 0:
                logger.info("Completed %d/%d basins", number, len(tasks))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary_path = out / "recession_basin_method_summary.csv"
    failure_path = out / "recession_failures.csv"
    combined.to_csv(summary_path, index=False)
    pd.DataFrame(failures, columns=["GAGE_ID", "error"]).to_csv(failure_path, index=False)
    if not frames:
        raise RuntimeError(f"No basin recession analysis succeeded; inspect {failure_path}")
    if failures:
        raise RuntimeError(f"{len(failures)} basins failed; rerun after inspecting {failure_path}")
    primary = combined[
        combined["method"].eq("p_screened_dry_state")
        & combined["tau_days"].eq(1)
    ] if not combined.empty else combined
    metrics = {
        "basins": len(frames),
        "forcing_screen_methods": int(combined["method"].nunique()) if not combined.empty else 0,
        "variance_exponent_available": int(primary["variance_exponent"].notna().sum()) if not primary.empty else 0,
        "strict_diagnostic_complete": int(primary["strict_diagnostics_complete"].sum()) if not primary.empty else 0,
        "finite_time_stochastic_screen_pass": int(
            primary["passes_finite_time_stochastic_screens"].sum()
        ) if not primary.empty else 0,
    }
    write_receipt(cfg, STAGE, [summary_path, failure_path, summary_dir, bins_dir], metrics)
    logger.info(
        "Stage 14 complete: conditional variance retained as a finite-time scaling relationship"
    )


if __name__ == "__main__":
    main()
