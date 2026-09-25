#!/usr/bin/env python3
"""Stage 08: audit temporal alignment, missingness, ranges, and primary eligibility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lib.common import (
    load_config, parse_args, prepare_stage, configure_logging, require_stage,
    stage_dir, write_receipt,
)


STAGE = 8


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def audit_file(row, cfg) -> dict:
    frame = pd.read_csv(row.file, low_memory=False)
    variables = ["q_mm_day", "pr_mm", "pet_mm", "tmin_c", "tmax_c", "tmean_c"]
    for column in variables:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    joint = frame[variables].notna().all(axis=1)
    valid_temp = frame["tmin_c"].le(frame["tmax_c"]) | frame[["tmin_c", "tmax_c"]].isna().any(axis=1)
    q_available = frame["q_mm_day"].notna()
    q_negative_fraction = float(frame.loc[q_available, "q_mm_day"].lt(0).mean()) if q_available.any() else np.nan
    pr_negative = int(frame["pr_mm"].lt(0).sum())
    pet_negative = int(frame["pet_mm"].lt(0).sum())
    cells = pd.to_numeric(frame["n_grid_cells"], errors="coerce").dropna()
    n_cells = int(cells.median()) if len(cells) else 0
    methods = frame["extraction_method"].dropna().astype(str)
    method = methods.mode().iloc[0] if len(methods) else "missing"
    joint_days = int(joint.sum())
    calendar_joint_fraction = float(joint.mean())
    if q_available.any():
        first_q = int(np.flatnonzero(q_available.to_numpy())[0])
        last_q = int(np.flatnonzero(q_available.to_numpy())[-1])
        record_span_days = last_q - first_q + 1
        joint_fraction = float(joint.iloc[first_q : last_q + 1].mean())
    else:
        record_span_days, joint_fraction = 0, np.nan
    eligible = (
        joint_days >= cfg["data_qc"]["minimum_joint_days"]
        and joint_fraction >= cfg["data_qc"]["minimum_joint_fraction"]
        and (not np.isfinite(q_negative_fraction) or q_negative_fraction <= cfg["data_qc"]["maximum_negative_q_fraction"])
        and (cfg["data_qc"]["allow_negative_precipitation"] or pr_negative == 0)
        and (cfg["data_qc"]["allow_negative_pet"] or pet_negative == 0)
        and bool(valid_temp.all())
        and n_cells >= cfg["gridmet"]["minimum_center_cells_for_primary_analysis"]
        and method == "center_in_polygon"
    )
    return {
        "GAGE_ID": str(row.GAGE_ID).zfill(8), "basin_id": row.basin_id,
        "quality_tier": row.quality_tier, "calendar_days": len(frame),
        "q_days": int(q_available.sum()), "pr_days": int(frame["pr_mm"].notna().sum()),
        "pet_days": int(frame["pet_mm"].notna().sum()), "temperature_days": int(frame["tmean_c"].notna().sum()),
        "joint_days": joint_days, "record_span_days": record_span_days,
        "joint_fraction_within_q_record": joint_fraction,
        "calendar_joint_fraction": calendar_joint_fraction,
        "negative_q_fraction": q_negative_fraction, "negative_pr_days": pr_negative,
        "negative_pet_days": pet_negative, "temperature_order_errors": int((~valid_temp).sum()),
        "n_grid_cells": n_cells, "extraction_method": method,
        "primary_eligible": eligible, "file": row.file,
    }


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 7)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    manifest = pd.read_csv(stage_dir(cfg, 7) / "matched_manifest.csv", dtype={"GAGE_ID": str})
    if args.limit:
        manifest = manifest.head(args.limit)
    rows = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        rows.append(audit_file(row, cfg))
        if number % 250 == 0:
            logger.info("Audited %d/%d basins", number, len(manifest))
    audit = pd.DataFrame(rows)
    audit_path = out / "matched_data_qc.csv"
    audit.to_csv(audit_path, index=False)
    eligible = audit[audit["primary_eligible"]].copy()
    eligible_path = out / "eligible_basins.csv"
    eligible.to_csv(eligible_path, index=False)
    metrics = {
        "audited_basins": len(audit), "primary_eligible": len(eligible),
        "excluded": len(audit) - len(eligible),
    }
    write_receipt(cfg, STAGE, [audit_path, eligible_path], metrics)
    logger.info("Stage 08 complete: %s primary-eligible basins", f"{len(eligible):,}")


if __name__ == "__main__":
    main()
