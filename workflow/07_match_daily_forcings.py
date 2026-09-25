#!/usr/bin/env python3
"""Stage 07: outer-match daily Q, precipitation, PET, and temperature for each basin."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from lib.common import (
    atomic_target, database_path, load_config, normalize_gage_id, parse_args,
    prepare_stage, configure_logging, require_stage, stage_dir, write_receipt,
)
from lib.database import connect
from lib.hydrology import cfs_to_mm_day


STAGE = 7


QUERY = """
WITH dates AS (
    SELECT date FROM discharge WHERE gage_id=?
    UNION SELECT date FROM precipitation WHERE gage_id=?
    UNION SELECT date FROM pet WHERE gage_id=?
    UNION SELECT date FROM temperature WHERE gage_id=?
)
SELECT dates.date, d.q_cfs, d.qualifier, p.pr_mm, e.pet_mm,
       t.tmin_c, t.tmax_c, t.tmean_c, t.valid_weight_fraction,
       t.n_grid_cells, t.extraction_method
FROM dates
LEFT JOIN discharge d ON d.gage_id=? AND d.date=dates.date
LEFT JOIN precipitation p ON p.gage_id=? AND p.date=dates.date
LEFT JOIN pet e ON e.gage_id=? AND e.date=dates.date
LEFT JOIN temperature t ON t.gage_id=? AND t.date=dates.date
WHERE dates.date BETWEEN ? AND ?
ORDER BY dates.date
"""


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 6)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    basins = pd.read_csv(stage_dir(cfg, 6) / "basin_index.csv", dtype={"GAGE_ID": str})
    basins = basins[basins["in_accepted_q_tier"].astype(str).str.lower().isin(["true", "1"])].copy()
    if args.limit:
        basins = basins.head(args.limit)
    daily_dir = out / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    connection = connect(database_path(cfg), readonly=True)
    manifest_rows = []
    try:
        for number, basin in enumerate(basins.itertuples(index=False), start=1):
            gage = normalize_gage_id(basin.GAGE_ID)
            parameters = [gage] * 8 + [cfg["project"]["start_date"], cfg["project"]["end_date"]]
            frame = pd.read_sql_query(QUERY, connection, params=parameters)
            if frame.empty:
                logger.warning("No records in database for %s", gage)
                continue
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            full_index = pd.date_range(cfg["project"]["start_date"], cfg["project"]["end_date"], freq="D")
            frame = frame.drop_duplicates("date").set_index("date").reindex(full_index).rename_axis("date").reset_index()
            frame.insert(0, "GAGE_ID", gage)
            frame.insert(0, "basin_id", str(basin.basin_id))
            frame["q_mm_day"] = cfs_to_mm_day(frame["q_cfs"], float(basin.area_km2))
            path = daily_dir / f"{gage}.csv.gz"
            with atomic_target(path) as temporary:
                frame.to_csv(temporary, index=False, compression="gzip")
            manifest_rows.append({
                "GAGE_ID": gage, "basin_id": basin.basin_id, "area_km2": basin.area_km2,
                "quality_tier": basin.quality_tier, "rows": len(frame), "file": str(path.resolve()),
            })
            if number % 100 == 0:
                logger.info("Matched %d/%d basins", number, len(basins))
    finally:
        connection.close()
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out / "matched_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    metrics = {"requested_basins": len(basins), "matched_basins": len(manifest), "daily_files": len(manifest)}
    write_receipt(cfg, STAGE, [manifest_path, daily_dir], metrics)
    logger.info("Stage 07 complete: %s matched basin files", f"{len(manifest):,}")


if __name__ == "__main__":
    main()
