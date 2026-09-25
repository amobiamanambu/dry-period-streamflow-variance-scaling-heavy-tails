"""GridMET inspection, basin weights, and temperature extraction.

The extraction uses cosine-latitude cell-area weights on the native regular grid.
Cells whose centers fall inside a basin are primary. Basins with no center cell can
use an explicitly flagged ``all_touched`` fallback; those basins are excluded from
the primary analysis by the downstream QC stage.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .common import normalize_gage_id


@dataclass(frozen=True)
class GridSpec:
    latitude: np.ndarray
    longitude: np.ndarray
    latitude_order: np.ndarray
    longitude_order: np.ndarray
    latitude_name: str
    longitude_name: str
    time_name: str
    data_name: str

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.latitude), len(self.longitude)


def _coordinate_name(dataset, candidates: tuple[str, ...], axis: str) -> str:
    lower = {name.lower(): name for name in dataset.variables}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    for name, variable in dataset.variables.items():
        if str(getattr(variable, "axis", "")).upper() == axis:
            return name
    raise ValueError(f"No {axis}-axis coordinate found in {dataset.filepath()}")


def inspect_grid(path: str | Path, variable_hint: str | None = None) -> GridSpec:
    from netCDF4 import Dataset

    with Dataset(path) as dataset:
        lat_name = _coordinate_name(dataset, ("lat", "latitude"), "Y")
        lon_name = _coordinate_name(dataset, ("lon", "longitude"), "X")
        time_name = _coordinate_name(dataset, ("day", "time"), "T")
        excluded = {lat_name, lon_name, time_name, "crs"}
        possible = []
        for name, variable in dataset.variables.items():
            dimensions = set(variable.dimensions)
            if name not in excluded and {lat_name, lon_name, time_name}.issubset(dimensions):
                possible.append(name)
        if variable_hint and variable_hint in possible:
            data_name = variable_hint
        elif len(possible) == 1:
            data_name = possible[0]
        elif "air_temperature" in possible:
            data_name = "air_temperature"
        else:
            raise ValueError(f"Ambiguous data variables in {path}: {possible}")

        raw_lat = np.asarray(dataset.variables[lat_name][:], dtype=float)
        raw_lon = np.asarray(dataset.variables[lon_name][:], dtype=float)
        canonical_lon = np.where(raw_lon > 180.0, raw_lon - 360.0, raw_lon)
        lat_order = np.argsort(raw_lat)[::-1]
        lon_order = np.argsort(canonical_lon)
        return GridSpec(
            latitude=raw_lat[lat_order],
            longitude=canonical_lon[lon_order],
            latitude_order=lat_order,
            longitude_order=lon_order,
            latitude_name=lat_name,
            longitude_name=lon_name,
            time_name=time_name,
            data_name=data_name,
        )


def grid_fingerprint(spec: GridSpec) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(spec.latitude, dtype="<f8").tobytes())
    digest.update(np.asarray(spec.longitude, dtype="<f8").tobytes())
    return digest.hexdigest()


def validate_netcdf(path: Path, expected_year: int, variable_hint: str | None = None) -> dict:
    from netCDF4 import Dataset, num2date

    spec = inspect_grid(path, variable_hint)
    with Dataset(path) as dataset:
        time_var = dataset.variables[spec.time_name]
        calendar = getattr(time_var, "calendar", "standard")
        dates = num2date(time_var[:], units=time_var.units, calendar=calendar)
        years = np.asarray([item.year for item in dates], dtype=int)
        variable = dataset.variables[spec.data_name]
        units = str(getattr(variable, "units", ""))
        size = int(variable.size)
        dimensions = list(variable.dimensions)
    if len(dates) not in (365, 366):
        raise ValueError(f"{path.name}: expected 365/366 days, found {len(dates)}")
    if not np.all(years == expected_year):
        raise ValueError(f"{path.name}: time coordinate is not confined to {expected_year}")
    if not ("k" in units.lower() or "c" in units.lower() or "degree" in units.lower()):
        raise ValueError(f"{path.name}: unrecognized temperature units {units!r}")
    return {
        "file": str(path.resolve()),
        "year": expected_year,
        "days": len(dates),
        "first_date": str(dates[0])[:10],
        "last_date": str(dates[-1])[:10],
        "data_variable": spec.data_name,
        "dimensions": dimensions,
        "values": size,
        "units": units,
        "grid_shape": list(spec.shape),
        "grid_fingerprint": grid_fingerprint(spec),
    }


def build_weight_matrix(
    shapefile: Path,
    spec: GridSpec,
    fallback: str = "all_touched",
):
    """Return a CSR basin-by-cell weight matrix and basin metadata."""
    import geopandas as gpd
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    from scipy import sparse

    basins = gpd.read_file(shapefile)
    if basins.crs is None:
        raise ValueError(f"Basin polygons have no CRS: {shapefile}")
    basins = basins.to_crs("EPSG:4326")
    if "GAGE_ID" not in basins.columns:
        raise ValueError(f"GAGE_ID is absent from {shapefile}")
    if "basin_id" not in basins.columns:
        basins["basin_id"] = [f"BASIN_{index + 1:05d}" for index in range(len(basins))]

    dlon = float(np.median(np.diff(spec.longitude)))
    dlat = float(abs(np.median(np.diff(spec.latitude))))
    if not np.allclose(np.diff(spec.longitude), dlon, rtol=0, atol=1e-6):
        raise ValueError("Longitude grid is not regular")
    if not np.allclose(np.diff(spec.latitude), -dlat, rtol=0, atol=1e-6):
        raise ValueError("Latitude grid is not regular descending")
    transform = from_origin(
        float(spec.longitude.min() - dlon / 2),
        float(spec.latitude.max() + dlat / 2),
        dlon,
        dlat,
    )
    cell_area_factor = np.cos(np.deg2rad(spec.latitude))[:, None]
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    records: list[dict] = []
    shape = spec.shape

    for basin_row, (_, basin) in enumerate(basins.iterrows()):
        geometry = basin.geometry
        if geometry is None or geometry.is_empty:
            records.append({
                "matrix_row": basin_row, "basin_id": basin["basin_id"],
                "gage_id": normalize_gage_id(basin["GAGE_ID"]), "n_grid_cells": 0,
                "extraction_method": "invalid_geometry",
            })
            continue
        mask = geometry_mask([geometry], transform=transform, out_shape=shape, invert=True, all_touched=False)
        method = "center_in_polygon"
        if not mask.any() and fallback == "all_touched":
            mask = geometry_mask([geometry], transform=transform, out_shape=shape, invert=True, all_touched=True)
            method = "all_touched_fallback"
        row_idx, col_idx = np.where(mask)
        weights = cell_area_factor[row_idx, 0]
        flat = row_idx * shape[1] + col_idx
        rows.extend([basin_row] * len(flat))
        cols.extend(flat.tolist())
        values.extend(weights.tolist())
        records.append({
            "matrix_row": basin_row,
            "basin_id": str(basin["basin_id"]),
            "gage_id": normalize_gage_id(basin["GAGE_ID"]),
            "n_grid_cells": int(len(flat)),
            "extraction_method": method if len(flat) else "no_grid_cell",
        })
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(basins), shape[0] * shape[1]))
    metadata = pd.DataFrame.from_records(records)
    return matrix, metadata


def _read_temperature_chunk(dataset, spec: GridSpec, start: int, stop: int) -> tuple[np.ndarray, str]:
    variable = dataset.variables[spec.data_name]
    selectors = []
    for dimension in variable.dimensions:
        if dimension == spec.time_name:
            selectors.append(slice(start, stop))
        elif dimension in (spec.latitude_name, spec.longitude_name):
            selectors.append(slice(None))
        else:
            raise ValueError(f"Unexpected dimension {dimension!r} in {spec.data_name}")
    raw = variable[tuple(selectors)]
    data = np.ma.filled(raw, np.nan).astype(float)
    axes = {name: index for index, name in enumerate(variable.dimensions)}
    data = np.moveaxis(
        data,
        (axes[spec.time_name], axes[spec.latitude_name], axes[spec.longitude_name]),
        (0, 1, 2),
    )
    data = data[:, spec.latitude_order, :][:, :, spec.longitude_order]
    units = str(getattr(variable, "units", ""))
    if "kelvin" in units.lower() or units.strip().lower() in {"k", "degk"} or np.nanmedian(data) > 100:
        data -= 273.15
    return data, units


def extract_temperature_chunks(
    tmin_path: Path,
    tmax_path: Path,
    weights,
    metadata: pd.DataFrame,
    chunk_days: int,
) -> Iterator[pd.DataFrame]:
    """Yield long basin-day frames without loading a full year into memory."""
    from netCDF4 import Dataset, num2date

    min_spec = inspect_grid(tmin_path)
    max_spec = inspect_grid(tmax_path)
    if grid_fingerprint(min_spec) != grid_fingerprint(max_spec):
        raise ValueError(f"tmmn/tmmx grids differ: {tmin_path.name}, {tmax_path.name}")
    if weights.shape[1] != min_spec.shape[0] * min_spec.shape[1]:
        raise ValueError("Saved weight matrix does not match the temperature grid")

    with Dataset(tmin_path) as minimum, Dataset(tmax_path) as maximum:
        min_time = minimum.variables[min_spec.time_name]
        max_time = maximum.variables[max_spec.time_name]
        min_dates = num2date(min_time[:], min_time.units, getattr(min_time, "calendar", "standard"))
        max_dates = num2date(max_time[:], max_time.units, getattr(max_time, "calendar", "standard"))
        min_keys = [f"{date.year:04d}-{date.month:02d}-{date.day:02d}" for date in min_dates]
        max_keys = [f"{date.year:04d}-{date.month:02d}-{date.day:02d}" for date in max_dates]
        if min_keys != max_keys:
            raise ValueError(f"tmmn/tmmx dates do not match for {tmin_path.name}")
        row_weight = np.asarray(weights.sum(axis=1)).ravel()
        n_basins = len(metadata)

        for start in range(0, len(min_keys), chunk_days):
            stop = min(start + chunk_days, len(min_keys))
            tmin, _ = _read_temperature_chunk(minimum, min_spec, start, stop)
            tmax, _ = _read_temperature_chunk(maximum, max_spec, start, stop)
            invalid_order = np.isfinite(tmin) & np.isfinite(tmax) & (tmin > tmax)
            tmin[invalid_order] = np.nan
            tmax[invalid_order] = np.nan

            def aggregate(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                flat = data.reshape(data.shape[0], -1)
                valid = np.isfinite(flat)
                numerator = (weights @ np.where(valid, flat, 0.0).T).T
                denominator = (weights @ valid.astype(float).T).T
                means = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
                fraction = np.divide(
                    denominator, row_weight[None, :], out=np.zeros_like(denominator), where=row_weight[None, :] > 0
                )
                return means, fraction

            minimum_mean, minimum_fraction = aggregate(tmin)
            maximum_mean, maximum_fraction = aggregate(tmax)
            valid_fraction = np.minimum(minimum_fraction, maximum_fraction)
            days = stop - start
            frame = pd.DataFrame({
                "basin_id": np.tile(metadata["basin_id"].to_numpy(), days),
                "GAGE_ID": np.tile(metadata["gage_id"].to_numpy(), days),
                "date": np.repeat(min_keys[start:stop], n_basins),
                "tmin_c": minimum_mean.reshape(-1),
                "tmax_c": maximum_mean.reshape(-1),
                "valid_weight_fraction": valid_fraction.reshape(-1),
                "n_grid_cells": np.tile(metadata["n_grid_cells"].to_numpy(), days),
                "extraction_method": np.tile(metadata["extraction_method"].to_numpy(), days),
            })
            frame["tmean_c"] = (frame["tmin_c"] + frame["tmax_c"]) / 2.0
            yield frame
