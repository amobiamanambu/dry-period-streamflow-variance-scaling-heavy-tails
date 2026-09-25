"""Estimators for statistical time irreversibility and stochastic recession.

The KL statistic implemented here is a path-asymmetry measure in nats. It is not
identified with physical thermodynamic entropy production because daily discharge
alone does not specify heat, work, microscopic states, or local detailed balance.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def regular_log_flow(frame: pd.DataFrame, maximum_gap_days: int = 3) -> pd.Series:
    """Return a regular daily log1p(Q) series with only short internal gaps filled."""
    dates = pd.to_datetime(frame["date"], errors="coerce")
    q = pd.Series(pd.to_numeric(frame["q_mm_day"], errors="coerce").to_numpy(), index=dates)
    q[q < 0] = np.nan
    q = q[~q.index.duplicated(keep="first")].sort_index()
    if q.empty:
        return q
    full = q.reindex(pd.date_range(q.index.min(), q.index.max(), freq="D"))
    logq = np.log1p(full)
    return logq.interpolate(limit=maximum_gap_days, limit_area="inside")


def deseasonalized_log_flow(frame: pd.DataFrame, maximum_gap_days: int = 3) -> pd.Series:
    """Remove the calendar-day median climatology from regular daily log flow."""
    logq = regular_log_flow(frame, maximum_gap_days)
    if logq.empty:
        return logq
    keys = pd.Index(logq.index.strftime("%m-%d"))
    climatology = logq.groupby(keys).median()
    anomalies = logq.to_numpy() - climatology.reindex(keys).to_numpy()
    return pd.Series(anomalies, index=logq.index, name="deseasonalized_logq")


def harmonic_deseasonalized_log_flow(
    frame: pd.DataFrame, maximum_gap_days: int = 3, harmonics: int = 3
) -> pd.Series:
    """Remove a smooth annual harmonic climatology as a seasonality sensitivity test."""
    logq = regular_log_flow(frame, maximum_gap_days)
    if logq.empty:
        return logq
    day = np.arange(len(logq), dtype=float)
    columns = [np.ones(len(logq), dtype=float)]
    for order in range(1, int(harmonics) + 1):
        angle = 2.0 * np.pi * order * day / 365.2425
        columns.extend([np.sin(angle), np.cos(angle)])
    design = np.column_stack(columns)
    values = logq.to_numpy(dtype=float)
    finite = np.isfinite(values)
    residual = np.full(len(values), np.nan, dtype=float)
    if finite.sum() > design.shape[1]:
        coefficient, *_ = np.linalg.lstsq(design[finite], values[finite], rcond=None)
        residual[finite] = values[finite] - design[finite] @ coefficient
    return pd.Series(residual, index=logq.index, name="harmonic_deseasonalized_logq")


def longest_finite_segment(series: pd.Series, minimum_length: int = 500) -> pd.Series:
    """Return the longest uninterrupted finite segment of a regular daily series."""
    finite = np.isfinite(series.to_numpy(dtype=float))
    if not finite.any():
        return series.iloc[0:0]
    changes = np.diff(np.r_[False, finite, False].astype(int))
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0]
    lengths = stops - starts
    best = int(np.argmax(lengths))
    if lengths[best] < minimum_length:
        return series.iloc[0:0]
    return series.iloc[starts[best] : stops[best]]


def quantile_states(values: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    states = np.full(len(x), -1, dtype=int)
    if finite.sum() < n_bins * 2:
        return states
    edges = np.unique(np.quantile(x[finite], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return states
    states[finite] = np.searchsorted(edges[1:-1], x[finite], side="right")
    return states


def transition_counts(states: np.ndarray, lag: int, n_bins: int) -> np.ndarray:
    origin = states[:-lag]
    destination = states[lag:]
    valid = (origin >= 0) & (destination >= 0)
    return np.bincount(
        origin[valid] * n_bins + destination[valid], minlength=n_bins * n_bins
    ).reshape(n_bins, n_bins)


def kl_path_asymmetry(counts: np.ndarray, pseudocount: float = 0.5) -> float:
    regularized = counts.astype(float) + pseudocount
    probability = regularized / regularized.sum()
    return float(np.sum(probability * np.log(probability / probability.T)))


def pairwise_irreversibility_decomposition(
    counts: np.ndarray, pseudocount: float = 0.5, tail_fraction: float = 0.2
) -> dict:
    """Decompose KL path asymmetry into nonnegative unordered state-pair terms.

    For each i < j the contribution is
    ``(p_ij - p_ji) log(p_ij / p_ji)``.  The terms sum exactly to the
    forward-versus-reverse KL statistic.  State labels are quantiles of the
    deseasonalized log-flow anomaly, so the tail fractions describe low/high
    *anomaly states*, not physical low/high discharge thresholds.
    """
    regularized = counts.astype(float) + pseudocount
    probability = regularized / regularized.sum()
    n_bins = probability.shape[0]
    tail_bins = max(1, int(np.ceil(n_bins * float(tail_fraction))))
    low, high, cross, total = 0.0, 0.0, 0.0, 0.0
    upward_current = 0.0
    for origin in range(n_bins):
        for destination in range(origin + 1, n_bins):
            forward = probability[origin, destination]
            reverse = probability[destination, origin]
            contribution = float((forward - reverse) * np.log(forward / reverse))
            total += contribution
            if origin < tail_bins or destination < tail_bins:
                low += contribution
            if origin >= n_bins - tail_bins or destination >= n_bins - tail_bins:
                high += contribution
            if origin < tail_bins and destination >= n_bins - tail_bins:
                cross += contribution
            upward_current += float(forward - reverse)
    if total <= 0:
        low_fraction = high_fraction = cross_fraction = np.nan
    else:
        low_fraction = low / total
        high_fraction = high / total
        cross_fraction = cross / total
    return {
        "pairwise_kl_total": float(total),
        "low_anomaly_pair_fraction": float(low_fraction),
        "high_anomaly_pair_fraction": float(high_fraction),
        "cross_tail_pair_fraction": float(cross_fraction),
        "net_upward_probability_current": float(upward_current),
    }


def irreversibility_spectrum(
    values: np.ndarray, lags: Iterable[int], n_bins: int, pseudocount: float = 0.5
) -> pd.DataFrame:
    states = quantile_states(values, n_bins)
    rows = []
    for lag in lags:
        counts = transition_counts(states, int(lag), n_bins)
        asymmetry = kl_path_asymmetry(counts, pseudocount)
        decomposition = pairwise_irreversibility_decomposition(counts, pseudocount)
        rows.append({
            "lag_days": int(lag),
            "n_transitions": int(counts.sum()),
            "irreversibility_nats": asymmetry,
            "irreversibility_rate_nats_day": asymmetry / int(lag),
            "occupied_pairs": int((counts > 0).sum()),
            **decomposition,
        })
    return pd.DataFrame(rows)


def iaaft_surrogate(values: np.ndarray, rng: np.random.Generator, iterations: int = 200) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("IAAFT input must be finite and regularly sampled")
    centered = x - x.mean()
    amplitude = np.abs(np.fft.rfft(centered))
    sorted_x = np.sort(x)
    surrogate = rng.permutation(x)
    for _ in range(iterations):
        phase = np.angle(np.fft.rfft(surrogate - surrogate.mean()))
        spectral = np.fft.irfft(amplitude * np.exp(1j * phase), n=len(x))
        ranks = np.argsort(np.argsort(spectral, kind="mergesort"), kind="mergesort")
        updated = sorted_x[ranks]
        if np.array_equal(updated, surrogate):
            break
        surrogate = updated
    return surrogate


def surrogate_corrected_spectrum(
    values: np.ndarray,
    lags: Iterable[int],
    n_bins: int,
    n_surrogates: int,
    iterations: int,
    pseudocount: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    x = np.asarray(values, dtype=float)
    if np.isnan(x).any():
        raise ValueError("Surrogate spectrum requires a complete regular series")
    observed = irreversibility_spectrum(x, lags, n_bins, pseudocount)
    null = np.empty((n_surrogates, len(observed)), dtype=float)
    for index in range(n_surrogates):
        surrogate = iaaft_surrogate(x, rng, iterations)
        null[index] = irreversibility_spectrum(surrogate, lags, n_bins, pseudocount)[
            "irreversibility_nats"
        ]
    null_mean = null.mean(axis=0)
    null_sd = null.std(axis=0, ddof=1) if n_surrogates > 1 else np.full(len(observed), np.nan)
    observed["iaaft_null_mean"] = null_mean
    observed["iaaft_null_sd"] = null_sd
    observed["irreversibility_bias_corrected"] = observed["irreversibility_nats"] - null_mean
    observed["iaaft_z"] = np.divide(
        observed["irreversibility_nats"] - null_mean,
        null_sd,
        out=np.full(len(observed), np.nan),
        where=null_sd > 0,
    )
    observed["iaaft_p_upper"] = (
        1 + (null >= observed["irreversibility_nats"].to_numpy()[None, :]).sum(axis=0)
    ) / (n_surrogates + 1)
    return observed


def ordinal_irreversibility(values: np.ndarray, dimension: int = 3, delay: int = 1) -> dict:
    x = np.asarray(values, dtype=float)
    patterns = list(itertools.permutations(range(dimension)))
    lookup = {pattern: index for index, pattern in enumerate(patterns)}
    forward = np.zeros(len(patterns), dtype=float)
    backward = np.zeros(len(patterns), dtype=float)
    span = (dimension - 1) * delay
    for start in range(0, len(x) - span):
        window = x[start : start + span + 1 : delay]
        if not np.all(np.isfinite(window)):
            continue
        pattern = tuple(np.argsort(window, kind="mergesort"))
        reverse_pattern = tuple(np.argsort(window[::-1], kind="mergesort"))
        forward[lookup[pattern]] += 1
        backward[lookup[reverse_pattern]] += 1
    p = (forward + 0.5) / (forward.sum() + 0.5 * len(forward))
    q = (backward + 0.5) / (backward.sum() + 0.5 * len(backward))
    return {
        "ordinal_dimension": dimension,
        "ordinal_delay_days": delay,
        "ordinal_patterns": int(forward.sum()),
        "ordinal_kl_nats": float(np.sum(p * np.log(p / q))),
    }


def transition_probability(values: np.ndarray, lag: int, n_bins: int) -> tuple[np.ndarray, int]:
    states = quantile_states(values, n_bins)
    counts = transition_counts(states, lag, n_bins).astype(float)
    row_sum = counts.sum(axis=1, keepdims=True)
    probability = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)
    return probability, int(counts.sum())


def chapman_kolmogorov_error(values: np.ndarray, lag: int, n_bins: int) -> dict:
    p1, n1 = transition_probability(values, lag, n_bins)
    p2, n2 = transition_probability(values, 2 * lag, n_bins)
    predicted = p1 @ p1
    denominator = np.linalg.norm(p2)
    error = np.linalg.norm(p2 - predicted) / denominator if denominator > 0 else np.nan
    return {"ck_lag_days": lag, "ck_error": float(error), "ck_n1": n1, "ck_n2": n2}


def conditional_kramers_moyal(
    q: np.ndarray,
    dq: np.ndarray,
    tau_days: int,
    n_bins: int,
    minimum_count: int,
) -> pd.DataFrame:
    q = np.asarray(q, dtype=float)
    dq = np.asarray(dq, dtype=float)
    valid = np.isfinite(q) & np.isfinite(dq) & (q > 0)
    q, dq = q[valid], dq[valid]
    if len(q) < n_bins * minimum_count:
        return pd.DataFrame()
    edges = np.unique(np.quantile(q, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 4:
        return pd.DataFrame()
    assignments = np.searchsorted(edges[1:-1], q, side="right")
    rows = []
    for state in range(len(edges) - 1):
        selected = assignments == state
        if selected.sum() < minimum_count:
            continue
        increments = dq[selected]
        mean_increment = float(np.mean(increments))
        first = mean_increment / tau_days
        centered = increments - mean_increment
        variance_rate = float(np.mean(centered**2) / (2 * tau_days))
        raw_second_rate = float(np.mean(increments**2) / (2 * tau_days))
        centered_fourth_rate = float(np.mean(centered**4) / (24 * tau_days))
        raw_fourth_rate = float(np.mean(increments**4) / (24 * tau_days))
        finite_time_ratio = (
            float(centered_fourth_rate / (variance_rate**2 * tau_days))
            if variance_rate > 0 else np.nan
        )
        rows.append({
            "state_bin": state,
            "q_center": float(np.exp(np.mean(np.log(q[selected])))),
            "n": int(selected.sum()),
            "D1": first,
            "conditional_mean_increment": mean_increment,
            "conditional_variance_rate": variance_rate,
            "D2_km_raw": raw_second_rate,
            "centered_fourth_rate": centered_fourth_rate,
            "D4_km_raw": raw_fourth_rate,
            "km4_finite_time_ratio": finite_time_ratio,
            # A locally Gaussian increment distribution has ratio 0.5. This is
            # a finite-time adequacy check, not by itself a Pawula-limit proof.
            "km4_gaussian_deviation": abs(finite_time_ratio - 0.5) if np.isfinite(finite_time_ratio) else np.nan,
        })
    return pd.DataFrame(rows)


def loglog_power_fit(x: np.ndarray, y: np.ndarray, minimum_points: int = 6) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if valid.sum() < minimum_points:
        return {"exponent": np.nan, "coefficient": np.nan, "r2": np.nan, "n_bins_fit": int(valid.sum())}
    lx, ly = np.log(x[valid]), np.log(y[valid])
    slope, intercept, r, p, stderr = stats.linregress(lx, ly)
    return {
        "exponent": float(slope),
        "coefficient": float(np.exp(intercept)),
        "r2": float(r**2),
        "p_value": float(p),
        "slope_se": float(stderr),
        "n_bins_fit": int(valid.sum()),
    }


def year_block_bootstrap_exponent(
    transitions: pd.DataFrame,
    tau_days: int,
    n_bins: int,
    minimum_count: int,
    minimum_points: int,
    replicates: int,
    rng: np.random.Generator,
) -> dict:
    work = transitions.copy()
    work["year"] = pd.to_datetime(work["date"]).dt.year
    years = np.asarray(sorted(work["year"].dropna().unique()))
    estimates = []
    if len(years) < 5:
        return {
            "bootstrap_valid": 0,
            "variance_exponent_ci_low": np.nan,
            "variance_exponent_ci_high": np.nan,
        }
    groups = {year: work[work["year"] == year] for year in years}
    for _ in range(replicates):
        sampled = rng.choice(years, size=len(years), replace=True)
        sample = pd.concat([groups[year] for year in sampled], ignore_index=True)
        km = conditional_kramers_moyal(
            sample["q_norm"], sample[f"dq_norm_{tau_days}d"], tau_days, n_bins, minimum_count
        )
        if km.empty:
            continue
        fit = loglog_power_fit(
            km["q_center"], km["conditional_variance_rate"], minimum_points
        )
        if np.isfinite(fit["exponent"]):
            estimates.append(fit["exponent"])
    if not estimates:
        return {
            "bootstrap_valid": 0,
            "variance_exponent_ci_low": np.nan,
            "variance_exponent_ci_high": np.nan,
        }
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "bootstrap_valid": len(estimates),
        "variance_exponent_ci_low": float(low),
        "variance_exponent_ci_high": float(high),
    }
