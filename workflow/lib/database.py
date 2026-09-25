"""SQLite schema and streaming insert/query utilities."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Iterator, Sequence


def connect(path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    target = Path(path).resolve()
    if readonly:
        connection = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=120)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(target, timeout=120)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=120000")
    if not readonly:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")
    return connection


def initialize_climate_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS precipitation (
            gage_id TEXT NOT NULL,
            date TEXT NOT NULL,
            pr_mm REAL,
            basin_id TEXT,
            PRIMARY KEY (gage_id, date)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS pet (
            gage_id TEXT NOT NULL,
            date TEXT NOT NULL,
            pet_mm REAL,
            basin_id TEXT,
            PRIMARY KEY (gage_id, date)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS temperature (
            gage_id TEXT NOT NULL,
            date TEXT NOT NULL,
            tmin_c REAL,
            tmax_c REAL,
            tmean_c REAL,
            valid_weight_fraction REAL,
            n_grid_cells INTEGER,
            extraction_method TEXT,
            basin_id TEXT,
            PRIMARY KEY (gage_id, date)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS source_ingest (
            source TEXT PRIMARY KEY,
            rows_inserted INTEGER NOT NULL,
            completed_utc TEXT NOT NULL,
            source_path TEXT NOT NULL
        );
        """
    )
    connection.commit()


def initialize_discharge_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS discharge (
            gage_id TEXT NOT NULL,
            date TEXT NOT NULL,
            q_cfs REAL,
            qualifier TEXT,
            source_file TEXT,
            PRIMARY KEY (gage_id, date)
        ) WITHOUT ROWID;
        """
    )
    connection.commit()


def replace_source(connection: sqlite3.Connection, table: str) -> None:
    allowed = {"precipitation", "pet", "temperature", "discharge"}
    if table not in allowed:
        raise ValueError(f"Refusing to replace unknown table: {table}")
    connection.execute(f"DELETE FROM {table}")
    connection.commit()


def batched_insert(
    connection: sqlite3.Connection,
    sql: str,
    rows: Iterable[Sequence[object]],
    batch_size: int = 50_000,
) -> int:
    count = 0
    batch: list[Sequence[object]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            connection.executemany(sql, batch)
            connection.commit()
            count += len(batch)
            batch.clear()
    if batch:
        connection.executemany(sql, batch)
        connection.commit()
        count += len(batch)
    return count


def table_count(connection: sqlite3.Connection, table: str) -> int:
    allowed = {"precipitation", "pet", "temperature", "discharge", "source_ingest"}
    if table not in allowed:
        raise ValueError(table)
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def iter_query(
    connection: sqlite3.Connection, sql: str, parameters: Sequence[object] = (), batch_size: int = 50_000
) -> Iterator[tuple]:
    cursor = connection.execute(sql, parameters)
    while rows := cursor.fetchmany(batch_size):
        yield from rows
