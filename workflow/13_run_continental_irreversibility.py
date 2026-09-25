#!/usr/bin/env python3
"""Stage 13: estimate resumable multiscale statistical time asymmetry."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from lib.analysis import InsufficientDataError, analyze_irreversibility
from lib.common import (
    atomic_json, atomic_target, load_config, parse_args, prepare_stage, configure_logging,
    read_json, require_stage, stage_dir, write_receipt,
)


STAGE = 13


def worker(task):
    gage, source_file, output_file, cfg, force = task
    output = Path(output_file)
    if output.exists() and output.with_suffix(".summary.json").exists() and not force:
        # The basin spectrum was already written atomically. Loading only its
        # compact summary makes a continental resume much faster.
        summary = read_json(output.with_suffix(".summary.json"))
        if int(summary.get("analysis_version", 0)) >= 4:
            return gage, summary, None
    frame = pd.read_csv(source_file, parse_dates=["date"], low_memory=False)
    rng = np.random.default_rng(cfg["project"]["random_seed"] + int(gage))
    spectrum, summary = analyze_irreversibility(
        frame, cfg, rng, cfg["irreversibility"]["iaaft_surrogates_continental"], False
    )
    spectrum.insert(0, "GAGE_ID", gage)
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_target(output) as temporary:
        spectrum.to_csv(temporary, index=False)
    atomic_json(output.with_suffix(".summary.json"), {"GAGE_ID": gage, **summary})
    return gage, {"GAGE_ID": gage, **summary}, None


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 12)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    inventory = pd.read_csv(stage_dir(cfg, 10) / "transition_inventory.csv", dtype={"GAGE_ID": str})
    if args.limit:
        inventory = inventory.head(args.limit)
    basin_dir = out / "by_basin"
    tasks = [
        (str(row.GAGE_ID).zfill(8), row.file, str(basin_dir / f"{str(row.GAGE_ID).zfill(8)}.csv"), cfg, args.force)
        for row in inventory.itertuples(index=False)
    ]
    summaries, exclusions, failures = [], [], []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(worker, task): task[0] for task in tasks}
        for number, future in enumerate(as_completed(futures), start=1):
            gage = futures[future]
            try:
                _, summary, _ = future.result()
                summaries.append(summary)
            except InsufficientDataError as error:
                exclusions.append({
                    "GAGE_ID": gage,
                    "status": "excluded_insufficient_data",
                    "reason": str(error),
                })
                logger.warning(
                    "Excluding %s from irreversibility analysis: %s", gage, error
                )
            except Exception as error:
                failures.append({"GAGE_ID": gage, "error": repr(error)})
                logger.error("Irreversibility failed for %s: %r", gage, error)
            if number % 100 == 0:
                logger.info("Completed %d/%d basins", number, len(tasks))
    summary_path = out / "irreversibility_basin_summary.csv"
    exclusion_path = out / "irreversibility_exclusions.csv"
    failure_path = out / "irreversibility_failures.csv"
    status_path = out / "irreversibility_basin_status.csv"
    if summaries:
        summary_frame = pd.DataFrame(summaries).sort_values("GAGE_ID")
        summary_frame.to_csv(summary_path, index=False)
    else:
        summary_frame = pd.DataFrame(columns=["GAGE_ID"])
        pd.DataFrame(columns=["GAGE_ID"]).to_csv(summary_path, index=False)
    exclusion_frame = pd.DataFrame(
        exclusions, columns=["GAGE_ID", "status", "reason"]
    ).sort_values("GAGE_ID")
    exclusion_frame.to_csv(exclusion_path, index=False)
    failure_frame = pd.DataFrame(
        failures, columns=["GAGE_ID", "error"]
    ).sort_values("GAGE_ID")
    failure_frame.to_csv(failure_path, index=False)
    status_rows = [
        {"GAGE_ID": str(gage).zfill(8), "status": "eligible", "reason": ""}
        for gage in summary_frame["GAGE_ID"]
    ]
    status_rows.extend(exclusions)
    status_rows.extend({
        "GAGE_ID": row["GAGE_ID"],
        "status": "failed_unexpected",
        "reason": row["error"],
    } for row in failures)
    status_frame = pd.DataFrame(
        status_rows, columns=["GAGE_ID", "status", "reason"]
    ).sort_values("GAGE_ID")
    status_frame.to_csv(status_path, index=False)
    if not summaries:
        raise RuntimeError(f"No basin irreversibility analysis succeeded; inspect {failure_path}")
    # Prespecified data-sufficiency exclusions are valid sample attrition, not
    # software failures. Unexpected exceptions remain fatal and block Stage 14.
    if failures:
        raise RuntimeError(f"{len(failures)} basins failed; rerun after inspecting {failure_path}")
    metrics = {
        "basins_requested": len(tasks),
        "basins_eligible": len(summaries),
        "excluded_insufficient_data": len(exclusions),
        "failed_unexpected": len(failures),
        "iaaft_surrogates_per_basin": int(
            cfg["irreversibility"]["iaaft_surrogates_continental"]
        ),
        "iaaft_p_resolution": float(
            1.0 / (cfg["irreversibility"]["iaaft_surrogates_continental"] + 1)
        ),
        "physical_entropy_production_claimed": False,
    }
    write_receipt(
        cfg, STAGE,
        [summary_path, exclusion_path, failure_path, status_path, basin_dir],
        metrics,
    )
    logger.info(
        "Stage 13 complete: %s eligible basin spectra; %s insufficient-data exclusions",
        f"{len(summaries):,}", f"{len(exclusions):,}",
    )


if __name__ == "__main__":
    main()
