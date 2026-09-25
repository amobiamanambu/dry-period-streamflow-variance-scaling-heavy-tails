#!/usr/bin/env python3
"""Stage 03: extract cell-area-weighted daily basin tmin, tmax, and tmean."""

from __future__ import annotations

import gzip

import pandas as pd
from scipy import sparse

from lib.common import (
    atomic_target, load_config, parse_args, prepare_stage, configure_logging,
    require_stage, resolve_path, stage_dir, write_receipt, year_range,
)
from lib.gridmet import build_weight_matrix, extract_temperature_chunks, inspect_grid


STAGE = 3


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 2)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    years = list(year_range(cfg))
    if args.limit:
        years = years[: args.limit]
    first_tmin = stage_dir(cfg, 1) / "netcdf" / "tmmn" / cfg["gridmet"]["variables"]["tmmn"].format(year=years[0])
    spec = inspect_grid(first_tmin)
    weights, metadata = build_weight_matrix(
        resolve_path(cfg, cfg["inputs"]["basin_polygons"]), spec,
        cfg["gridmet"]["small_basin_fallback"],
    )
    if args.limit:
        # --limit applies to both years and basins at this stage for a true small smoke run.
        metadata = metadata.head(args.limit).copy()
        weights = weights[: len(metadata)]
    matrix_path = out / "basin_grid_weights.npz"
    metadata_path = out / "basin_grid_weight_metadata.csv"
    sparse.save_npz(matrix_path, weights)
    metadata.to_csv(metadata_path, index=False)
    outputs = [matrix_path, metadata_path]
    logger.info("Built weights for %d basins (%d non-zero basin-cell links)", weights.shape[0], weights.nnz)

    for year in years:
        tmin_path = stage_dir(cfg, 1) / "netcdf" / "tmmn" / cfg["gridmet"]["variables"]["tmmn"].format(year=year)
        tmax_path = stage_dir(cfg, 1) / "netcdf" / "tmmx" / cfg["gridmet"]["variables"]["tmmx"].format(year=year)
        annual_path = out / "annual" / f"basin_temperature_{year}.csv.gz"
        annual_path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_target(annual_path) as temporary:
            with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
                header = True
                for chunk in extract_temperature_chunks(
                    tmin_path, tmax_path, weights, metadata, cfg["gridmet"]["chunk_days"]
                ):
                    low_support = chunk["valid_weight_fraction"] < cfg["gridmet"]["minimum_valid_weight_fraction"]
                    chunk.loc[low_support, ["tmin_c", "tmax_c", "tmean_c"]] = float("nan")
                    chunk.to_csv(handle, index=False, header=header)
                    header = False
        outputs.append(annual_path)
        logger.info("Extracted basin temperature for %d", year)
    metrics = {
        "basins": len(metadata), "years": years, "basin_cell_links": int(weights.nnz),
        "no_cell_basins": int((metadata["n_grid_cells"] == 0).sum()),
        "fallback_basins": int((metadata["extraction_method"] == "all_touched_fallback").sum()),
    }
    write_receipt(cfg, STAGE, outputs, metrics)
    logger.info("Stage 03 complete")


if __name__ == "__main__":
    main()
