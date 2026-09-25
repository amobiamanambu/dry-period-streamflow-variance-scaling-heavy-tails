"""Streaming parser for multi-section USGS daily-value RDB gzip files."""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from .common import normalize_gage_id


Q_PATTERN = re.compile(r"00060_00003")


def parse_rdb_gzip(path: Path) -> Iterator[tuple[str, str, float | None, str, str]]:
    """Yield ``gage_id, date, q_cfs, qualifier, source_file`` records.

    Historical parameter time-series identifiers can change. For each row, this
    parser coalesces all mean-daily-discharge columns from left to right and retains
    the associated USGS qualifier. Duplicate gage-date resolution is performed by
    the database primary key in stage 05 and reported in its audit table.
    """
    columns: list[str] = []
    qcols: list[int] = []
    qualifier_for: dict[int, int] = {}
    skip_type_row = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("agency_cd"):
                columns = line.rstrip("\n").split("\t")
                qcols = [
                    index for index, name in enumerate(columns)
                    if Q_PATTERN.search(name) and not name.endswith("_cd")
                ]
                qualifier_for = {
                    index: columns.index(f"{columns[index]}_cd")
                    for index in qcols if f"{columns[index]}_cd" in columns
                }
                skip_type_row = True
                continue
            if skip_type_row:
                skip_type_row = False
                continue
            if not columns or not qcols:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gage_id = normalize_gage_id(parts[1])
            date = parts[2].strip()[:10]
            value: float | None = None
            qualifier = ""
            for index in qcols:
                if index >= len(parts) or not parts[index].strip():
                    continue
                try:
                    candidate = float(parts[index])
                except ValueError:
                    continue
                if np.isfinite(candidate):
                    value = candidate
                    qindex = qualifier_for.get(index)
                    qualifier = parts[qindex].strip() if qindex is not None and qindex < len(parts) else ""
                    break
            yield gage_id, date, value, qualifier, path.name


def manifest_site_file_map(manifest_path: Path) -> dict[str, str]:
    import pandas as pd

    manifest = pd.read_csv(manifest_path, dtype=str)
    if "sites" not in manifest or "raw_file" not in manifest:
        raise ValueError("Discharge manifest must contain sites and raw_file columns")
    mapping: dict[str, str] = {}
    for row in manifest.itertuples(index=False):
        for site in str(row.sites).split(","):
            normalized = normalize_gage_id(site)
            if normalized:
                mapping[normalized] = str(row.raw_file)
    return mapping
