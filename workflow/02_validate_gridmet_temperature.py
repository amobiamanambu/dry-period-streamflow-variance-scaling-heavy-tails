#!/usr/bin/env python3
"""Stage 02: validate GridMET time axes, units, grids, and annual completeness."""

from __future__ import annotations

import pandas as pd

from lib.common import (
    atomic_json, load_config, parse_args, prepare_stage, configure_logging,
    require_stage, stage_dir, write_receipt, year_range,
)
from lib.gridmet import validate_netcdf


STAGE = 2


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 1)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    records = []
    years = list(year_range(cfg))
    if args.limit:
        years = years[: args.limit]
    for variable, template in cfg["gridmet"]["variables"].items():
        for year in years:
            path = stage_dir(cfg, 1) / "netcdf" / variable / template.format(year=year)
            if not path.exists():
                raise FileNotFoundError(f"Missing GridMET input: {path}")
            record = validate_netcdf(path, year)
            record["gridmet_code"] = variable
            records.append(record)
            logger.info("Validated %s", path.name)
    audit = pd.DataFrame(records).sort_values(["gridmet_code", "year"])
    for year, group in audit.groupby("year"):
        if group["grid_fingerprint"].nunique() != 1:
            raise RuntimeError(f"tmmn and tmmx grids differ in {year}")
        if group["days"].nunique() != 1:
            raise RuntimeError(f"tmmn and tmmx day counts differ in {year}")
    if audit["grid_fingerprint"].nunique() != 1:
        raise RuntimeError("Grid changes across annual files; extraction weights cannot be safely reused")
    csv_path = out / "temperature_netcdf_audit.csv"
    audit.to_csv(csv_path, index=False)
    summary = {
        "files": len(audit), "years": sorted(audit["year"].unique().tolist()),
        "grid_fingerprint": audit["grid_fingerprint"].iloc[0],
        "grid_shape": audit["grid_shape"].iloc[0],
    }
    json_path = out / "temperature_netcdf_summary.json"
    atomic_json(json_path, summary)
    write_receipt(cfg, STAGE, [csv_path, json_path], summary)
    logger.info("Stage 02 complete: all temperature files share one validated grid")


if __name__ == "__main__":
    main()
