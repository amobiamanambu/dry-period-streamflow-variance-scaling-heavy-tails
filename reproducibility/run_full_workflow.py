#!/usr/bin/env python3
"""Guarded runner for the computationally intensive national workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / "workflow"
CONFIG = REPO_ROOT / "config" / "full_study.json"
STAGES = {
    0: "00_check_environment.py",
    1: "01_download_gridmet_temperature.py",
    2: "02_validate_gridmet_temperature.py",
    3: "03_extract_basin_temperature.py",
    4: "04_build_climate_database.py",
    5: "05_ingest_usgs_discharge.py",
    6: "06_build_basin_index.py",
    7: "07_match_daily_forcings.py",
    8: "08_quality_control_matched_data.py",
    9: "09_build_snowmelt_proxy.py",
    10: "10_build_transition_inventory.py",
    11: "11_select_validation_basins.py",
    12: "12_validate_estimators.py",
    13: "13_run_continental_irreversibility.py",
    14: "14_run_continental_recession.py",
    15: "15_assemble_continental_results.py",
    16: "16_test_hypotheses.py",
    17: "17_detect_temporal_change.py",
    18: "18_test_operational_value.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--from-stage", type=int, default=0, choices=STAGES)
    parser.add_argument("--to-stage", type=int, default=18, choices=STAGES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-sensitivity-analyses",
        action="store_true",
        help="After Stage 18, run every versioned sensitivity analysis used in the study.",
    )
    args = parser.parse_args()
    if not args.confirm_full_run:
        raise SystemExit(
            "Full reconstruction can require about 150 GB and one to two days. "
            "Read docs/FULL_WORKFLOW.md and rerun with --confirm-full-run."
        )
    if args.from_stage > args.to_stage:
        raise SystemExit("--from-stage must not exceed --to-stage")
    if args.include_sensitivity_analyses and args.to_stage < 18:
        raise SystemExit("Sensitivity analyses require a completed run through Stage 18")

    for number in range(args.from_stage, args.to_stage + 1):
        script = WORKFLOW / STAGES[number]
        command = [sys.executable, str(script), "--config", str(CONFIG)]
        if number > 0:
            command.extend(["--workers", str(max(1, args.workers))])
            if args.force:
                command.append("--force")
        print(f"\n[stage {number:02d}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    if args.include_sensitivity_analyses:
        command = [
            sys.executable,
            str(REPO_ROOT / "reproducibility" / "run_sensitivity_analyses.py"),
            "--config",
            str(CONFIG),
            "--workers",
            str(max(1, args.workers)),
        ]
        if args.force:
            command.append("--force")
        print(f"\n[sensitivity analyses] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
