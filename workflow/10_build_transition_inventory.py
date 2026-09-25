#!/usr/bin/env python3
"""Stage 10: construct lagged Q increments and forcing-screened recession flags."""

from __future__ import annotations

import pandas as pd

from lib.analysis import transition_method_masks
from lib.common import (
    atomic_target, load_config, parse_args, prepare_stage, configure_logging,
    require_stage, stage_dir, write_receipt,
)
from lib.hydrology import add_transition_columns


STAGE = 10


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 9)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    manifest = pd.read_csv(stage_dir(cfg, 9) / "snow_daily_manifest.csv", dtype={"GAGE_ID": str})
    if args.limit:
        manifest = manifest.head(args.limit)
    daily_dir = out / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        frame = pd.read_csv(row.file, parse_dates=["date"], low_memory=False)
        transitions = add_transition_columns(frame, cfg["transition_definitions"])
        counts = {}
        for tau in cfg["transition_definitions"]["recession_lags_days"]:
            masks = transition_method_masks(transitions, cfg, tau)
            for method, mask in masks.items():
                column = f"select_{method}_{tau}d"
                transitions[column] = mask
                counts[column] = int(mask.sum())
        path = daily_dir / f"{str(row.GAGE_ID).zfill(8)}.csv.gz"
        with atomic_target(path) as temporary:
            transitions.to_csv(temporary, index=False, compression="gzip")
        rows.append({
            "GAGE_ID": str(row.GAGE_ID).zfill(8), "basin_id": row.basin_id,
            "file": str(path.resolve()), **counts,
        })
        if number % 100 == 0:
            logger.info("Constructed transitions for %d/%d basins", number, len(manifest))
    inventory_path = out / "transition_inventory.csv"
    pd.DataFrame(rows).to_csv(inventory_path, index=False)
    write_receipt(cfg, STAGE, [inventory_path, daily_dir], {"basins": len(rows)})
    logger.info("Stage 10 complete")


if __name__ == "__main__":
    main()
