#!/usr/bin/env python3
"""Stage 06: assemble one authoritative CONUS basin index and GAGES-II attributes."""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

from lib.common import (
    load_config, normalize_gage_id, parse_args, prepare_stage, configure_logging,
    require_stage, resolve_path, write_receipt,
)


STAGE = 6


def read_zip_table(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    frame = pd.read_csv(archive.open(name), dtype={"STAID": str}, encoding="latin1", low_memory=False)
    frame["STAID"] = frame["STAID"].map(normalize_gage_id)
    return frame


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 5)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    inventory = pd.read_csv(resolve_path(cfg, cfg["inputs"]["basin_inventory"]), dtype={"GAGE_ID": str})
    inventory["GAGE_ID"] = inventory["GAGE_ID"].map(normalize_gage_id)
    qc = pd.read_csv(resolve_path(cfg, cfg["inputs"]["discharge_qc"]), dtype={"GAGE_ID": str}, low_memory=False)
    qc["GAGE_ID"] = qc["GAGE_ID"].map(normalize_gage_id)
    zip_path = resolve_path(cfg, cfg["inputs"]["gagesii_attributes_zip"])
    with zipfile.ZipFile(zip_path) as archive:
        names = [
            "conterm_basinid.txt", "conterm_bas_classif.txt", "conterm_bas_morph.txt",
            "conterm_climate.txt", "conterm_hydro.txt", "conterm_flowrec.txt",
        ]
        tables = [read_zip_table(archive, name) for name in names]
    attributes = tables[0]
    for table in tables[1:]:
        shared = [column for column in table.columns if column in attributes.columns and column != "STAID"]
        attributes = attributes.merge(table.drop(columns=shared), on="STAID", how="left", validate="one_to_one")
    index = inventory.merge(attributes, left_on="GAGE_ID", right_on="STAID", how="left", validate="one_to_one")
    qc_keep = [column for column in qc.columns if column not in index.columns or column == "GAGE_ID"]
    index = index.merge(qc[qc_keep], on="GAGE_ID", how="left", validate="one_to_one")
    index["area_km2"] = pd.to_numeric(index["area_km2"], errors="coerce")
    index["log_area"] = np.log10(index["area_km2"].where(index["area_km2"] > 0))
    index["snow_fraction"] = pd.to_numeric(index.get("SNOW_PCT_PRECIP"), errors="coerce") / 100.0
    annual_precip_mm = pd.to_numeric(index.get("PPTAVG_BASIN"), errors="coerce") * 10.0
    index["aridity"] = pd.to_numeric(index.get("PET"), errors="coerce") / annual_precip_mm
    index["is_reference"] = index.get("CLASS", "").astype(str).str.lower().eq("ref")
    index["in_primary_q_tier"] = index["quality_tier"].eq(cfg["data_qc"]["primary_tier"])
    index["in_accepted_q_tier"] = index["quality_tier"].isin(cfg["data_qc"]["accepted_tiers"])
    if args.limit:
        index = index.head(args.limit)
    output = out / "basin_index.csv"
    index.to_csv(output, index=False)
    audit = {
        "inventory_basins": len(inventory), "indexed_basins": len(index),
        "attributes_matched": int(index["STAID"].notna().sum()),
        "accepted_q_tier": int(index["in_accepted_q_tier"].sum()),
        "primary_q_tier": int(index["in_primary_q_tier"].sum()),
        "reference_basins": int(index["is_reference"].sum()),
    }
    write_receipt(cfg, STAGE, [output], audit)
    logger.info("Stage 06 complete: %s indexed basins", f"{len(index):,}")


if __name__ == "__main__":
    main()
