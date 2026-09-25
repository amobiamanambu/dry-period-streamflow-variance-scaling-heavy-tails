#!/usr/bin/env python3
"""Stage 11: select a reproducible, hydroclimatically diverse estimator-validation sample."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lib.common import (
    load_config, parse_args, prepare_stage, configure_logging, require_stage,
    seed_everything, stage_dir, write_receipt,
)


STAGE = 11


def quantile_label(series: pd.Series, name: str, bins: int = 4) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(numeric, bins, labels=[f"{name}_q{i + 1}" for i in range(bins)], duplicates="drop").astype(str)
    except ValueError:
        return pd.Series(f"{name}_missing", index=series.index)


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 10)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    rng = seed_everything(cfg["project"]["random_seed"])
    index = pd.read_csv(stage_dir(cfg, 6) / "basin_index.csv", dtype={"GAGE_ID": str}, low_memory=False)
    snow = pd.read_csv(stage_dir(cfg, 9) / "snow_proxy_summary.csv", dtype={"GAGE_ID": str})
    transitions = pd.read_csv(stage_dir(cfg, 10) / "transition_inventory.csv", dtype={"GAGE_ID": str})
    candidates = transitions.merge(index, on=["GAGE_ID", "basin_id"], how="left").merge(
        snow[["GAGE_ID", "snowfall_fraction_of_pr"]], on="GAGE_ID", how="left"
    )
    candidates["snow_stratum"] = quantile_label(candidates["snowfall_fraction_of_pr"], "snow")
    candidates["bfi_stratum"] = quantile_label(candidates["BFI_AVE"], "bfi")
    candidates["aridity_stratum"] = quantile_label(candidates["aridity"], "aridity")
    candidates["area_stratum"] = quantile_label(candidates["log_area"], "area")
    candidates["ecoregion_stratum"] = candidates["AGGECOREGION"].fillna(candidates["ecoregion"]).fillna("unknown")
    candidates["class_stratum"] = candidates["CLASS"].fillna("unknown")
    candidates["base_stratum"] = (
        candidates["ecoregion_stratum"].astype(str) + "|" + candidates["class_stratum"].astype(str)
    )
    candidates["random_order"] = rng.random(len(candidates))
    size = min(cfg["validation"]["sample_size"], len(candidates))
    # Seed with one basin per ecoregion/class group, then use greedy maximin
    # sampling in rank-scaled snow, BFI, aridity, and area space.
    seeds = (
        candidates.sort_values("random_order").groupby("base_stratum", group_keys=False).head(1)
        .sort_values("random_order").head(size)
    )
    selected_indices = list(seeds.index)
    numeric_columns = ["snowfall_fraction_of_pr", "BFI_AVE", "aridity", "log_area"]
    coordinates = candidates[numeric_columns].apply(pd.to_numeric, errors="coerce")
    coordinates = coordinates.rank(pct=True).fillna(0.5).to_numpy(dtype=float)
    all_indices = list(candidates.index)
    index_to_position = {index: position for position, index in enumerate(all_indices)}
    remaining = set(all_indices) - set(selected_indices)
    while len(selected_indices) < size and remaining:
        selected_positions = [index_to_position[index] for index in selected_indices]
        best_index, best_distance = None, -np.inf
        for candidate_index in remaining:
            position = index_to_position[candidate_index]
            distance = np.sqrt(
                ((coordinates[selected_positions] - coordinates[position]) ** 2).sum(axis=1)
            ).min()
            distance += float(candidates.loc[candidate_index, "random_order"]) * 1e-9
            if distance > best_distance:
                best_index, best_distance = candidate_index, distance
        selected_indices.append(best_index)
        remaining.remove(best_index)
    selected = candidates.loc[selected_indices].copy()
    if args.limit:
        selected = selected.head(args.limit)
    selected["manual_hydrograph_inspection"] = False
    manual_size = min(cfg["validation"]["manual_inspection_size"], len(selected))
    selected.loc[selected.index[:manual_size], "manual_hydrograph_inspection"] = True
    output = out / "validation_basins.csv"
    selected.to_csv(output, index=False)
    strata = out / "validation_strata_counts.csv"
    selected.groupby(["ecoregion_stratum", "class_stratum"], dropna=False).size().rename("n").reset_index().to_csv(strata, index=False)
    metrics = {"candidate_basins": len(candidates), "selected_basins": len(selected), "manual_inspection": manual_size}
    write_receipt(cfg, STAGE, [output, strata], metrics)
    logger.info("Stage 11 complete: %d diverse validation basins", len(selected))


if __name__ == "__main__":
    main()
