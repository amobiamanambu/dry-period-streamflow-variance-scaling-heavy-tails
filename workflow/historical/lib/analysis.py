"""Basin-level analysis functions shared by validation and continental stages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hydrology import contiguous_run_filter
from .stochastic import (
    chapman_kolmogorov_error, conditional_kramers_moyal, deseasonalized_log_flow,
    irreversibility_spectrum, loglog_power_fit, longest_finite_segment,
    ordinal_irreversibility, surrogate_corrected_spectrum,
    year_block_bootstrap_exponent,
)


class InsufficientDataError(ValueError):
    """Raised when a basin cannot support a prespecified estimator."""


def analyze_irreversibility(
    frame: pd.DataFrame,
    cfg: dict,
    rng: np.random.Generator,
    n_surrogates: int,
    surrogates_all_bins: bool = False,
) -> tuple[pd.DataFrame, dict]:
    settings = cfg["irreversibility"]
    anomaly = deseasonalized_log_flow(frame)
    segment = longest_finite_segment(anomaly, max(settings["minimum_transition_count"], 500))
    if segment.empty:
        raise InsufficientDataError(
            "No sufficiently long complete daily segment for irreversibility"
        )
    values = segment.to_numpy(dtype=float)
    lags = [lag for lag in cfg["transition_definitions"]["lags_days"] if lag < len(values)]
    rows = []
    for n_bins in settings["state_bins"]:
        use_surrogates = surrogates_all_bins or n_bins == settings["primary_state_bins"]
        if use_surrogates:
            spectrum = surrogate_corrected_spectrum(
                values, lags, n_bins, n_surrogates, settings["iaaft_iterations"],
                settings["pseudocount"], rng,
            )
        else:
            spectrum = irreversibility_spectrum(values, lags, n_bins, settings["pseudocount"])
            for column in (
                "iaaft_null_mean", "iaaft_null_sd", "irreversibility_bias_corrected",
                "iaaft_z", "iaaft_p_upper",
            ):
                spectrum[column] = np.nan
        spectrum["state_bins"] = n_bins
        rows.append(spectrum)
    spectra = pd.concat(rows, ignore_index=True)
    primary = spectra[spectra["state_bins"] == settings["primary_state_bins"]].sort_values("lag_days")
    corrected = primary["irreversibility_bias_corrected"].clip(lower=0).to_numpy(dtype=float)
    if not np.isfinite(corrected).any():
        corrected = primary["irreversibility_nats"].to_numpy(dtype=float)
    lags_array = primary["lag_days"].to_numpy(dtype=float)
    peak_index = int(np.nanargmax(corrected))
    ordinal = ordinal_irreversibility(
        values, settings["ordinal_dimension"], settings["ordinal_delay_days"]
    )
    summary = {
        "analysis_version": 4,
        "segment_start": str(segment.index.min().date()),
        "segment_end": str(segment.index.max().date()),
        "segment_days": len(segment),
        "primary_state_bins": settings["primary_state_bins"],
        "irreversibility_1d": float(primary.loc[primary["lag_days"] == 1, "irreversibility_nats"].iloc[0])
        if (primary["lag_days"] == 1).any() else np.nan,
        "bias_corrected_1d": float(primary.loc[primary["lag_days"] == 1, "irreversibility_bias_corrected"].iloc[0])
        if (primary["lag_days"] == 1).any() else np.nan,
        "peak_lag_days": int(lags_array[peak_index]),
        "peak_bias_corrected": float(corrected[peak_index]),
        "spectrum_auc_loglag": float(np.trapezoid(corrected, np.log(lags_array))) if len(corrected) > 1 else np.nan,
        "significant_lags_0p05": int((primary["iaaft_p_upper"] <= 0.05).sum()),
        "lags_at_minimum_surrogate_p": int(
            (primary["iaaft_p_upper"] <= (1.0 / (n_surrogates + 1) + 1e-12)).sum()
        ),
        "iaaft_surrogates": int(n_surrogates),
        "iaaft_p_resolution": float(1.0 / (n_surrogates + 1)),
        "peak_lag_right_censored": bool(
            int(lags_array[peak_index]) == max(cfg["transition_definitions"]["lags_days"])
        ),
        "low_anomaly_pair_fraction_1d": float(
            primary.loc[primary["lag_days"] == 1, "low_anomaly_pair_fraction"].iloc[0]
        ) if (primary["lag_days"] == 1).any() else np.nan,
        "high_anomaly_pair_fraction_1d": float(
            primary.loc[primary["lag_days"] == 1, "high_anomaly_pair_fraction"].iloc[0]
        ) if (primary["lag_days"] == 1).any() else np.nan,
        "net_upward_probability_current_1d": float(
            primary.loc[
                primary["lag_days"] == 1, "net_upward_probability_current"
            ].iloc[0]
        ) if (primary["lag_days"] == 1).any() else np.nan,
        **ordinal,
    }
    return spectra, summary


def transition_method_specs(
    frame: pd.DataFrame, cfg: dict, tau: int, include_forcing_sensitivity: bool = False
) -> list[dict]:
    """Return prespecified transition masks and explicit forcing-screen metadata."""
    settings = cfg["transition_definitions"]
    valid = frame[f"valid_transition_{tau}d"].fillna(False).astype(bool)
    decline = frame[f"dq_norm_{tau}d"].le(0)
    base_threshold = settings["precipitation_thresholds_mm_day"][0]
    base_antecedent = 3 if 3 in settings["antecedent_windows_days"] else settings["antecedent_windows_days"][0]
    label = str(base_threshold).replace(".", "p")
    dry = frame[f"dry_p{label}_a{base_antecedent}_l{tau}"].fillna(False).astype(bool)
    minimum = settings["minimum_contiguous_days"]
    specifications = [
        {
            "method": "q_only_monotone",
            "mask": contiguous_run_filter(valid & decline, minimum),
            "screening_pr_threshold_mm": np.nan,
            "screening_antecedent_days": 0,
            "requires_monotone_decline": True,
            "is_primary_method": False,
        },
        {
            "method": "p_screened_monotone",
            "mask": contiguous_run_filter(valid & decline & dry, minimum),
            "screening_pr_threshold_mm": base_threshold,
            "screening_antecedent_days": base_antecedent,
            "requires_monotone_decline": True,
            "is_primary_method": False,
        },
        {
            "method": "p_screened_dry_state",
            "mask": contiguous_run_filter(valid & dry, minimum),
            "screening_pr_threshold_mm": base_threshold,
            "screening_antecedent_days": base_antecedent,
            "requires_monotone_decline": False,
            "is_primary_method": True,
        },
    ]
    if include_forcing_sensitivity:
        for threshold in settings["precipitation_thresholds_mm_day"]:
            threshold_label = str(threshold).replace(".", "p")
            for antecedent in settings["antecedent_windows_days"]:
                if threshold == base_threshold and antecedent == base_antecedent:
                    continue
                dry_column = f"dry_p{threshold_label}_a{antecedent}_l{tau}"
                sensitivity_dry = frame[dry_column].fillna(False).astype(bool)
                specifications.append({
                    "method": f"p_screened_dry_state_p{threshold_label}_a{antecedent}",
                    "mask": contiguous_run_filter(valid & sensitivity_dry, minimum),
                    "screening_pr_threshold_mm": threshold,
                    "screening_antecedent_days": antecedent,
                    "requires_monotone_decline": False,
                    "is_primary_method": False,
                })
    return specifications


def transition_method_masks(frame: pd.DataFrame, cfg: dict, tau: int) -> dict[str, np.ndarray]:
    """Compatibility wrapper used when only the three primary method masks are needed."""
    return {
        specification["method"]: specification["mask"]
        for specification in transition_method_specs(frame, cfg, tau, False)
    }


def analyze_recession(
    frame: pd.DataFrame,
    cfg: dict,
    rng: np.random.Generator,
    bootstrap_replicates: int,
    bootstrap_primary_only: bool = True,
    include_forcing_sensitivity: bool = False,
    store_sensitivity_bins: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    settings = cfg["recession"]
    summaries = []
    bin_frames = []
    q_all = pd.to_numeric(frame["q_norm"], errors="coerce").to_numpy(dtype=float)
    ck = {}
    for tau in cfg["transition_definitions"]["recession_lags_days"]:
        ck[tau] = chapman_kolmogorov_error(q_all, tau, settings["state_bins"])
        specifications = transition_method_specs(
            frame, cfg, tau, include_forcing_sensitivity
        )
        for specification in specifications:
            method, mask = specification["method"], specification["mask"]
            selected = frame.loc[mask].copy()
            km = conditional_kramers_moyal(
                selected["q_norm"], selected[f"dq_norm_{tau}d"], tau,
                settings["state_bins"], settings["minimum_bin_count"],
            )
            if not km.empty:
                km["method"] = method
                km["tau_days"] = tau
                if store_sensitivity_bins or method in {
                    "q_only_monotone", "p_screened_monotone", "p_screened_dry_state"
                }:
                    bin_frames.append(km)
                variance = loglog_power_fit(
                    km["q_center"], km["conditional_variance_rate"],
                    settings["minimum_valid_bins_for_fit"],
                )
                drift = loglog_power_fit(
                    km["q_center"], -km["D1"], settings["minimum_valid_bins_for_fit"]
                )
            else:
                variance = loglog_power_fit(
                    np.array([]), np.array([]), settings["minimum_valid_bins_for_fit"]
                )
                drift = variance.copy()
            bootstrap = {
                "bootstrap_valid": 0,
                "variance_exponent_ci_low": np.nan,
                "variance_exponent_ci_high": np.nan,
            }
            if (
                bootstrap_replicates > 0
                and (not bootstrap_primary_only or (method == "p_screened_dry_state" and tau == 1))
                and len(selected)
            ):
                bootstrap = year_block_bootstrap_exponent(
                    selected, tau, settings["state_bins"], settings["minimum_bin_count"],
                    settings["minimum_valid_bins_for_fit"], bootstrap_replicates, rng,
                )
            summaries.append({
                "method": method, "tau_days": tau, "n_transitions": int(mask.sum()),
                "screening_pr_threshold_mm": specification["screening_pr_threshold_mm"],
                "screening_antecedent_days": specification["screening_antecedent_days"],
                "requires_monotone_decline": specification["requires_monotone_decline"],
                "is_primary_method": specification["is_primary_method"],
                "variance_exponent": variance["exponent"],
                "variance_coefficient": variance["coefficient"],
                "variance_r2": variance["r2"], "variance_bins": variance["n_bins_fit"],
                "drift_exponent": drift["exponent"], "drift_coefficient": drift["coefficient"],
                "drift_r2": drift["r2"],
                "median_km4_finite_time_ratio": float(km["km4_finite_time_ratio"].median()) if not km.empty else np.nan,
                "median_km4_gaussian_deviation": float(km["km4_gaussian_deviation"].median()) if not km.empty else np.nan,
                **ck[tau], **bootstrap,
            })
    summary = pd.DataFrame(summaries)
    bins = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
    primary = summary[(summary["method"] == "p_screened_dry_state") & (summary["tau_days"] == 1)]
    variance_exponent = float(primary["variance_exponent"].iloc[0]) if len(primary) else np.nan
    variance_r2 = float(primary["variance_r2"].iloc[0]) if len(primary) else np.nan
    ck_error = float(primary["ck_error"].iloc[0]) if len(primary) else np.nan
    km4_deviation = (
        float(primary["median_km4_gaussian_deviation"].iloc[0]) if len(primary) else np.nan
    )
    markov_available = bool(np.isfinite(ck_error))
    km4_available = bool(np.isfinite(km4_deviation))
    markov_pass = bool(markov_available and ck_error <= settings["markov_tolerance"])
    km4_pass = bool(
        km4_available and km4_deviation <= settings["km4_gaussian_deviation_warning"]
    )
    diagnostic_complete = markov_available and km4_available
    diagnostic = {
        "primary_variance_exponent": variance_exponent,
        "primary_variance_r2": variance_r2,
        "primary_ck_error": ck_error,
        "primary_km4_gaussian_deviation": km4_deviation,
        "markov_diagnostic_available": markov_available,
        "km4_diagnostic_available": km4_available,
        "strict_diagnostics_complete": diagnostic_complete,
        "markov_pass": markov_pass,
        "km4_local_gaussian_pass": km4_pass,
        "passes_finite_time_stochastic_screens": bool(
            diagnostic_complete and markov_pass and km4_pass
        ),
        # Legacy alias retained for old Stage-14 files. These screens do not
        # identify an infinitesimal Gaussian diffusion from daily observations.
        "passes_strict_daily_diffusion_diagnostics": bool(
            diagnostic_complete and markov_pass and km4_pass
        ),
        "markov_warning": not markov_pass,
        "km4_warning": not km4_pass,
    }
    return summary, bins, diagnostic
