"""Hydrologic unit conversions, snow proxy, and transition construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


CFS_TO_MM_DAY_PER_KM2 = 2.446575545


def cfs_to_mm_day(q_cfs: pd.Series | np.ndarray, area_km2: float) -> np.ndarray:
    if not np.isfinite(area_km2) or area_km2 <= 0:
        return np.full(len(q_cfs), np.nan)
    return np.asarray(q_cfs, dtype=float) * CFS_TO_MM_DAY_PER_KM2 / area_km2


def precipitation_snow_fraction(
    tmean_c: pd.Series | np.ndarray, snow_temperature_c: float, rain_temperature_c: float
) -> np.ndarray:
    temperature = np.asarray(tmean_c, dtype=float)
    fraction = (rain_temperature_c - temperature) / (rain_temperature_c - snow_temperature_c)
    return np.clip(fraction, 0.0, 1.0)


def add_temperature_index_snow_proxy(frame: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Add a transparent degree-day snow-storage proxy.

    This is a forcing-screening proxy, not a reconstruction of SWE. Missing forcing
    breaks the storage recursion and restarts it at zero on the next valid day.
    """
    out = frame.sort_values("date").copy()
    phase = precipitation_snow_fraction(
        out["tmean_c"], settings["snow_temperature_c"], settings["rain_temperature_c"]
    )
    pr = out["pr_mm"].to_numpy(dtype=float)
    temperature = out["tmean_c"].to_numpy(dtype=float)
    snowfall = pr * phase
    potential_melt = settings["degree_day_factor_mm_c_day"] * np.maximum(
        temperature - settings["melt_temperature_c"], 0.0
    )
    storage_before = np.full(len(out), np.nan)
    storage_after = np.full(len(out), np.nan)
    melt = np.full(len(out), np.nan)
    storage = 0.0
    for index in range(len(out)):
        if not (np.isfinite(pr[index]) and np.isfinite(temperature[index])):
            storage = 0.0
            continue
        storage_before[index] = storage
        available = max(0.0, storage + max(0.0, snowfall[index]))
        melt[index] = min(available, max(0.0, potential_melt[index]))
        storage = max(0.0, available - melt[index])
        storage_after[index] = storage
    out["snow_fraction"] = phase
    out["snowfall_proxy_mm"] = snowfall
    out["snow_storage_before_mm"] = storage_before
    out["snowmelt_proxy_mm"] = melt
    out["snow_storage_after_mm"] = storage_after
    out["snowmelt_risk"] = (
        (out["snow_storage_before_mm"] >= settings["minimum_snow_storage_mm"])
        & (out["snowmelt_proxy_mm"] > 0)
    )
    return out


def add_transition_columns(frame: pd.DataFrame, settings: dict) -> pd.DataFrame:
    out = frame.sort_values("date").copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    q = out["q_mm_day"].to_numpy(dtype=float)
    median_q = float(np.nanmedian(q[q > settings["minimum_q_mm_day"]])) if np.any(q > 0) else np.nan
    out["q_norm"] = out["q_mm_day"] / median_q
    # Long-lag irreversibility is computed directly from Q, so only the short
    # lags needed by the recession estimator are materialized here.
    for lag in settings["recession_lags_days"]:
        consecutive = (dates.shift(-lag) - dates).dt.days.eq(lag)
        out[f"q_next_{lag}d"] = out["q_norm"].shift(-lag)
        out[f"dq_norm_{lag}d"] = out[f"q_next_{lag}d"] - out["q_norm"]
        out[f"valid_transition_{lag}d"] = (
            consecutive & out["q_norm"].gt(0) & out[f"q_next_{lag}d"].gt(0)
        )

    melt = out.get("snowmelt_risk", pd.Series(False, index=out.index)).fillna(True).astype(bool)
    for lag in settings["recession_lags_days"]:
        for antecedent in settings["antecedent_windows_days"]:
            # Window is t-(a-1), ..., t, ..., t+lag. This excludes transitions
            # with precipitation or proxy snowmelt anywhere over the increment.
            offsets = range(-(antecedent - 1), lag + 1)
            p_window = pd.concat([out["pr_mm"].shift(-offset) for offset in offsets], axis=1).max(
                axis=1, skipna=False
            )
            melt_components = pd.concat(
                [melt.shift(-offset) for offset in offsets], axis=1
            )
            # Conservatively mark a window as melt-affected if any day has melt
            # risk or if any required day is unavailable. Expressing this logic
            # directly avoids pandas' deprecated silent object-to-bool downcast.
            melt_window = melt_components.eq(True).any(axis=1) | melt_components.isna().any(axis=1)
            out[f"forcing_window_max_pr_a{antecedent}_l{lag}"] = p_window
            out[f"forcing_window_melt_a{antecedent}_l{lag}"] = melt_window
            for threshold in settings["precipitation_thresholds_mm_day"]:
                label = str(threshold).replace(".", "p")
                out[f"dry_p{label}_a{antecedent}_l{lag}"] = p_window.le(threshold) & ~melt_window
    return out


def contiguous_run_filter(flags: pd.Series, minimum_days: int) -> np.ndarray:
    values = flags.fillna(False).to_numpy(dtype=bool)
    selected = np.zeros(len(values), dtype=bool)
    start = None
    for index, flag in enumerate(np.r_[values, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= minimum_days:
                selected[start:index] = True
            start = None
    return selected
