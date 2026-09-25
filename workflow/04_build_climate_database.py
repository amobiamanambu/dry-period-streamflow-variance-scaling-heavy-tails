#!/usr/bin/env python3
"""Stage 04: standardize P, PET, and temperature into an indexed SQLite database."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lib.common import (
    database_path, load_config, normalize_gage_id, parse_args, prepare_stage,
    configure_logging, require_stage, resolve_path, stage_dir, utc_now, write_receipt,
)
from lib.database import batched_insert, connect, initialize_climate_schema, replace_source, table_count


STAGE = 4


def completed_source(connection, source: str) -> int | None:
    row = connection.execute(
        "SELECT rows_inserted FROM source_ingest WHERE source=?", (source,)
    ).fetchone()
    return int(row[0]) if row else None


def table_has_rows(connection, table: str) -> bool:
    allowed = {"precipitation", "pet", "temperature"}
    if table not in allowed:
        raise ValueError(table)
    return connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None


def climate_rows(path: Path, value_column: str, output_name: str, chunk_size: int = 500_000):
    for frame in pd.read_csv(path, dtype={"GAGE_ID": str, "basin_id": str}, chunksize=chunk_size):
        required = {"GAGE_ID", "date", value_column}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path.name} lacks columns {sorted(required - set(frame.columns))}")
        values = pd.to_numeric(frame[value_column], errors="coerce")
        basins = frame["basin_id"] if "basin_id" in frame else pd.Series("", index=frame.index)
        for gage, date, value, basin in zip(frame["GAGE_ID"], frame["date"], values, basins):
            yield normalize_gage_id(gage), str(date)[:10], float(value) if np.isfinite(value) else None, str(basin)


def nullable_float(values: pd.Series) -> list[float | None]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return [float(value) if np.isfinite(value) else None for value in numeric]


def temperature_rows(path: Path):
    """Yield database-ready rows using column arrays rather than pandas.iterrows."""
    for frame in pd.read_csv(path, dtype={"GAGE_ID": str, "basin_id": str}, chunksize=250_000):
        required = {
            "GAGE_ID", "date", "tmin_c", "tmax_c", "tmean_c", "valid_weight_fraction",
            "n_grid_cells", "extraction_method", "basin_id",
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"{path.name} lacks columns {sorted(required - set(frame.columns))}")
        gages = [normalize_gage_id(value) for value in frame["GAGE_ID"].to_numpy()]
        dates = frame["date"].astype(str).str[:10].tolist()
        tmin = nullable_float(frame["tmin_c"])
        tmax = nullable_float(frame["tmax_c"])
        tmean = nullable_float(frame["tmean_c"])
        valid_fraction = nullable_float(frame["valid_weight_fraction"])
        cells = pd.to_numeric(frame["n_grid_cells"], errors="coerce").fillna(0).astype(int).tolist()
        methods = frame["extraction_method"].fillna("missing").astype(str).tolist()
        basins = frame["basin_id"].fillna("").astype(str).tolist()
        yield from zip(gages, dates, tmin, tmax, tmean, valid_fraction, cells, methods, basins)


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 3)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    db_path = database_path(cfg)
    connection = connect(db_path)
    initialize_climate_schema(connection)
    sources = [
        ("precipitation", resolve_path(cfg, cfg["inputs"]["precipitation_csv"]), "precipitation_mm", "pr_mm"),
        ("pet", resolve_path(cfg, cfg["inputs"]["pet_csv"]), "pet_mm", "pet_mm"),
    ]
    counts = {}
    try:
        for table, path, input_column, _ in sources:
            if not path.exists():
                raise FileNotFoundError(path)
            if args.force:
                replace_source(connection, table)
                connection.execute("DELETE FROM source_ingest WHERE source=?", (table,))
                connection.commit()
            else:
                prior_count = completed_source(connection, table)
                if prior_count is not None:
                    counts[table] = prior_count
                    logger.info("Resuming stage: retained completed %s source (%s rows)", table, f"{prior_count:,}")
                    continue
                if table_has_rows(connection, table):
                    logger.warning("Removing incomplete partial %s table before resuming", table)
                    replace_source(connection, table)
            count = batched_insert(
                connection,
                f"INSERT OR REPLACE INTO {table} (gage_id,date,{table if table == 'pet' else 'pr_mm'},basin_id) VALUES (?,?,?,?)"
                if table == "precipitation" else
                "INSERT OR REPLACE INTO pet (gage_id,date,pet_mm,basin_id) VALUES (?,?,?,?)",
                climate_rows(path, input_column, table),
            )
            counts[table] = count
            connection.execute(
                "INSERT OR REPLACE INTO source_ingest VALUES (?,?,?,?)",
                (table, count, utc_now(), str(path.resolve())),
            )
            connection.commit()
            logger.info("Ingested %s %s source rows", f"{count:,}", table)

        temperature_paths = sorted((stage_dir(cfg, 3) / "annual").glob("basin_temperature_*.csv.gz"))
        if args.limit:
            temperature_paths = temperature_paths[: args.limit]
        if args.force:
            replace_source(connection, "temperature")
            connection.execute("DELETE FROM source_ingest WHERE source='temperature' OR source LIKE 'temperature:%'")
            connection.commit()
        prior_temperature_count = None if args.force else completed_source(connection, "temperature")
        if prior_temperature_count is not None:
            temp_count = prior_temperature_count
            logger.info(
                "Resuming stage: retained completed temperature source (%s rows)",
                f"{temp_count:,}",
            )
        else:
            if not args.force and table_has_rows(connection, "temperature"):
                logger.info(
                    "Retaining partial temperature rows; primary-key upserts will safely overwrite and continue"
                )
            temp_count = 0
            for annual_path in temperature_paths:
                year = annual_path.stem.split("_")[-1].split(".")[0]
                annual_key = f"temperature:{year}"
                prior_annual_count = None if args.force else completed_source(connection, annual_key)
                if prior_annual_count is not None:
                    temp_count += prior_annual_count
                    logger.info(
                        "Retained completed temperature year %s (%s rows)",
                        year, f"{prior_annual_count:,}",
                    )
                    continue
                logger.info("Ingesting temperature year %s from %s", year, annual_path.name)
                annual_count = batched_insert(
                    connection,
                    "INSERT OR REPLACE INTO temperature "
                    "(gage_id,date,tmin_c,tmax_c,tmean_c,valid_weight_fraction,n_grid_cells,extraction_method,basin_id) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    temperature_rows(annual_path),
                )
                connection.execute(
                    "INSERT OR REPLACE INTO source_ingest VALUES (?,?,?,?)",
                    (annual_key, annual_count, utc_now(), str(annual_path.resolve())),
                )
                connection.commit()
                temp_count += annual_count
                logger.info("Completed temperature year %s (%s rows)", year, f"{annual_count:,}")
        counts["temperature"] = temp_count
        if prior_temperature_count is None:
            connection.execute(
                "INSERT OR REPLACE INTO source_ingest VALUES (?,?,?,?)",
                ("temperature", temp_count, utc_now(), str((stage_dir(cfg, 3) / "annual").resolve())),
            )
            connection.commit()
        # Full COUNT(*) scans over three ~149-million-row WITHOUT ROWID tables
        # add hours without improving the primary-key integrity guarantee.
        audit = pd.DataFrame([
            {"table": name, "source_rows": count, "primary_key": "gage_id,date",
             "duplicate_policy": "INSERT OR REPLACE"}
            for name, count in counts.items()
        ])
        audit_path = out / "climate_ingest_audit.csv"
        audit.to_csv(audit_path, index=False)
    finally:
        connection.close()
    write_receipt(cfg, STAGE, [db_path, audit_path], counts)
    logger.info("Stage 04 complete: %s", db_path)


if __name__ == "__main__":
    main()
