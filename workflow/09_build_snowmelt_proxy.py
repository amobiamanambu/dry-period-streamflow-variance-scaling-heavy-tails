#!/usr/bin/env python3
"""Stage 09: add an explicitly approximate temperature-index snow accumulation/melt screen."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from lib.common import (
    atomic_target, load_config, parse_args, prepare_stage, configure_logging,
    require_stage, stage_dir, write_receipt,
)
from lib.hydrology import add_temperature_index_snow_proxy


STAGE = 9


def proxy_summary(gage: str, frame: pd.DataFrame, settings: dict, label: str) -> dict:
    annual_pr = float(frame["pr_mm"].sum(min_count=1))
    snowfall = float(frame["snowfall_proxy_mm"].sum(min_count=1))
    return {
        "GAGE_ID": gage, "parameter_set": label,
        "snow_temperature_c": settings["snow_temperature_c"],
        "rain_temperature_c": settings["rain_temperature_c"],
        "degree_day_factor": settings["degree_day_factor_mm_c_day"],
        "forcing_days": int(frame[["pr_mm", "tmean_c"]].notna().all(axis=1).sum()),
        "snowfall_proxy_mm_total": snowfall,
        "snowfall_fraction_of_pr": snowfall / annual_pr if annual_pr > 0 else np.nan,
        "snowmelt_proxy_mm_total": float(frame["snowmelt_proxy_mm"].sum(min_count=1)),
        "snowmelt_risk_days": int(frame["snowmelt_risk"].fillna(False).sum()),
        "maximum_storage_proxy_mm": float(frame["snow_storage_after_mm"].max()),
    }


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 8)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    eligible = pd.read_csv(stage_dir(cfg, 8) / "eligible_basins.csv", dtype={"GAGE_ID": str})
    if args.limit:
        eligible = eligible.head(args.limit)
    daily_dir = out / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    base_summaries, sensitivity_summaries, manifest_rows = [], [], []
    base = cfg["snow_proxy"].copy()
    combinations = itertools.product(
        base["sensitivity_snow_temperature_c"],
        base["sensitivity_rain_temperature_c"],
        base["sensitivity_degree_day_factor"],
    )
    combinations = [combo for combo in combinations if combo[0] < combo[1]]
    for number, row in enumerate(eligible.itertuples(index=False), start=1):
        gage = str(row.GAGE_ID).zfill(8)
        frame = pd.read_csv(row.file, parse_dates=["date"], low_memory=False)
        augmented = add_temperature_index_snow_proxy(frame, base)
        output_path = daily_dir / f"{gage}.csv.gz"
        with atomic_target(output_path) as temporary:
            augmented.to_csv(temporary, index=False, compression="gzip")
        base_summaries.append(proxy_summary(gage, augmented, base, "primary"))
        for snow_t, rain_t, factor in combinations:
            settings = base.copy()
            settings.update({
                "snow_temperature_c": snow_t, "rain_temperature_c": rain_t,
                "degree_day_factor_mm_c_day": factor,
            })
            sensitivity = add_temperature_index_snow_proxy(frame, settings)
            label = f"snow{snow_t:g}_rain{rain_t:g}_ddf{factor:g}"
            sensitivity_summaries.append(proxy_summary(gage, sensitivity, settings, label))
        manifest_rows.append({"GAGE_ID": gage, "basin_id": row.basin_id, "file": str(output_path.resolve())})
        if number % 100 == 0:
            logger.info("Built snow proxy for %d/%d basins", number, len(eligible))
    summary_path = out / "snow_proxy_summary.csv"
    sensitivity_path = out / "snow_proxy_sensitivity.csv"
    manifest_path = out / "snow_daily_manifest.csv"
    pd.DataFrame(base_summaries).to_csv(summary_path, index=False)
    pd.DataFrame(sensitivity_summaries).to_csv(sensitivity_path, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    metrics = {"basins": len(manifest_rows), "sensitivity_parameter_sets": len(combinations)}
    write_receipt(cfg, STAGE, [summary_path, sensitivity_path, manifest_path, daily_dir], metrics)
    logger.info("Stage 09 complete: proxy outputs are screening variables, not observed SWE")


if __name__ == "__main__":
    main()
