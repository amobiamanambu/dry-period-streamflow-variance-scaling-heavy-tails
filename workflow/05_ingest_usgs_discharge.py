#!/usr/bin/env python3
"""Stage 05: parse USGS daily-value RDB files into the continental SQLite database."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lib.common import (
    database_path, load_config, parse_args, prepare_stage, configure_logging,
    require_stage, resolve_path, sha256_file, success_path, utc_now,
    validate_stage_outputs, write_receipt,
)
from lib.database import batched_insert, connect, initialize_discharge_schema, replace_source, table_count
from lib.discharge import parse_rdb_gzip


STAGE = 5


def validate_existing_stage5_before_rerun(cfg):
    """Protect the shared climate/discharge database before a forced reload."""
    receipt_path = success_path(cfg, STAGE)
    if not receipt_path.exists():
        return False
    receipt = validate_stage_outputs(cfg, STAGE)
    if int(receipt.get("receipt_schema", 1)) < 2:
        raise RuntimeError(
            "The existing Stage-05 receipt predates content fingerprints. "
            "Rebuild Stages 04 and 05 rather than forcing an unverifiable database in place."
        )
    return True


def verify_download_manifest(files, manifest_path):
    """Verify selected USGS batches against the immutable download manifest."""
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {"raw_file", "bytes_gz", "sha256_gz"}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise ValueError(
            "Discharge manifest is missing required columns: "
            + ", ".join(missing_columns)
        )

    manifest = manifest.copy()
    manifest["raw_file"] = manifest["raw_file"].map(lambda value: Path(value).name)
    if (manifest["raw_file"] == "").any():
        raise ValueError("Discharge manifest contains an empty raw_file value")
    duplicates = sorted(
        manifest.loc[manifest["raw_file"].duplicated(keep=False), "raw_file"].unique()
    )
    if duplicates:
        raise ValueError(
            "Discharge manifest contains duplicate raw_file entries: "
            + ", ".join(duplicates[:5])
        )

    indexed = manifest.set_index("raw_file", drop=False)
    errors = []
    for path in files:
        if path.name not in indexed.index:
            errors.append(f"not listed in manifest: {path.name}")
            continue
        row = indexed.loc[path.name]
        try:
            expected_bytes = int(row["bytes_gz"])
        except (TypeError, ValueError):
            errors.append(f"invalid bytes_gz for {path.name}: {row['bytes_gz']!r}")
            continue
        if path.stat().st_size != expected_bytes:
            errors.append(
                f"byte count mismatch for {path.name}: "
                f"expected {expected_bytes}, found {path.stat().st_size}"
            )
            continue
        expected_sha256 = str(row["sha256_gz"]).strip().lower()
        observed_sha256 = sha256_file(path)
        if observed_sha256 != expected_sha256:
            errors.append(
                f"SHA-256 mismatch for {path.name}: "
                f"expected {expected_sha256}, found {observed_sha256}"
            )
    if errors:
        raise RuntimeError(
            "USGS discharge batch verification failed:\n- " + "\n- ".join(errors)
        )
    return manifest


def all_rows(files, logger):
    for index, path in enumerate(files, start=1):
        yield from parse_rdb_gzip(path)
        if index % 10 == 0:
            logger.info("Parsed %d/%d RDB gzip files", index, len(files))


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    # Stage 04 and Stage 05 intentionally share one SQLite file: Stage 05 adds
    # discharge tables to the climate database.  Validate the pristine Stage-04
    # file on the first run; on a forced Stage-05 rerun, validate the Stage-04
    # receipt chain and output existence without rejecting the legitimate
    # downstream mutation already captured by the Stage-05 fingerprint.
    existing_stage5 = success_path(cfg, STAGE).exists()
    if existing_stage5 and not args.force:
        raise RuntimeError(
            f"Stage {STAGE:02d} already completed at {success_path(cfg, STAGE)}. "
            "Use --force only if you intend to replace its products."
        )
    stage5_was_verified = (
        validate_existing_stage5_before_rerun(cfg) if existing_stage5 else False
    )
    require_stage(cfg, 4, validate_outputs=not stage5_was_verified)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    raw_directory = resolve_path(cfg, cfg["inputs"]["discharge_directory"])
    manifest_path = resolve_path(cfg, cfg["inputs"]["discharge_manifest"])
    all_files = sorted(raw_directory.glob("*.rdb.gz"))
    if not all_files:
        raise FileNotFoundError(f"No .rdb.gz files found in {raw_directory}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Discharge manifest not found: {manifest_path}")
    files = all_files
    if args.limit:
        files = files[: args.limit]
    manifest = verify_download_manifest(files, manifest_path)
    if args.limit is None:
        manifest_names = set(manifest["raw_file"])
        disk_names = {path.name for path in all_files}
        missing_on_disk = sorted(manifest_names - disk_names)
        unlisted_on_disk = sorted(disk_names - manifest_names)
        if missing_on_disk or unlisted_on_disk:
            messages = []
            if missing_on_disk:
                messages.append(
                    "manifest files missing on disk: " + ", ".join(missing_on_disk[:5])
                )
            if unlisted_on_disk:
                messages.append(
                    "disk files absent from manifest: " + ", ".join(unlisted_on_disk[:5])
                )
            raise RuntimeError("USGS discharge snapshot is incomplete: " + "; ".join(messages))
    logger.info(
        "Verified %d/%d USGS batches against %s",
        len(files), len(manifest), manifest_path.name,
    )
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
            "manifest_rows": len(manifest), "manifest_files_verified": len(files),
            "manifest_sha256": sha256_file(manifest_path),
        }])
        audit_path = out / "discharge_ingest_audit.csv"
        audit.to_csv(audit_path, index=False)
    finally:
        connection.close()
    write_receipt(cfg, STAGE, [db_path, audit_path], audit.iloc[0].to_dict())
    logger.info("Stage 05 complete: %s unique gage-days", f"{unique_rows:,}")


if __name__ == "__main__":
    main()
