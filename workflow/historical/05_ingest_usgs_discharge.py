#!/usr/bin/env python3
"""Stage 05: parse USGS daily-value RDB files into the continental SQLite database."""

from __future__ import annotations

import pandas as pd

from lib.common import (
    database_path, load_config, parse_args, prepare_stage, configure_logging,
    require_stage, resolve_path, utc_now, write_receipt,
)
from lib.database import batched_insert, connect, initialize_discharge_schema, replace_source, table_count
from lib.discharge import parse_rdb_gzip


STAGE = 5


def all_rows(files, logger):
    for index, path in enumerate(files, start=1):
        yield from parse_rdb_gzip(path)
        if index % 10 == 0:
            logger.info("Parsed %d/%d RDB gzip files", index, len(files))


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 4)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    raw_directory = resolve_path(cfg, cfg["inputs"]["discharge_directory"])
    files = sorted(raw_directory.glob("*.rdb.gz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No .rdb.gz files found in {raw_directory}")
    db_path = database_path(cfg)
    connection = connect(db_path)
    initialize_discharge_schema(connection)
    try:
        if args.force:
            replace_source(connection, "discharge")
        elif table_count(connection, "discharge"):
            raise RuntimeError("Database table discharge already has data; use --force to replace it")
        input_rows = batched_insert(
            connection,
            "INSERT OR REPLACE INTO discharge (gage_id,date,q_cfs,qualifier,source_file) VALUES (?,?,?,?,?)",
            all_rows(files, logger),
        )
        unique_rows = table_count(connection, "discharge")
        connection.execute(
            "INSERT OR REPLACE INTO source_ingest VALUES (?,?,?,?)",
            ("discharge", input_rows, utc_now(), str(raw_directory.resolve())),
        )
        connection.commit()
        audit = pd.DataFrame([{
            "source_files": len(files), "input_rows": input_rows, "unique_gage_dates": unique_rows,
            "duplicate_gage_dates_replaced": input_rows - unique_rows,
        }])
        audit_path = out / "discharge_ingest_audit.csv"
        audit.to_csv(audit_path, index=False)
    finally:
        connection.close()
    write_receipt(cfg, STAGE, [db_path, audit_path], audit.iloc[0].to_dict())
    logger.info("Stage 05 complete: %s unique gage-days", f"{unique_rows:,}")


if __name__ == "__main__":
    main()
