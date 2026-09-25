#!/usr/bin/env python3
"""Versioned sidecar sensitivity for signed tail multiples and basin attributes.

This script reads the receipt-verified signed-tail basin archive and the local
GAGES-II CONUS attribute release.  It does not read daily data, refit the tail
model or modify the numbered workflow stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy import stats

from lib.common import PROJECT_ROOT, atomic_json, atomic_target, sha256_file, utc_now
from lib.statistics import benjamini_hochberg


ANALYSIS_NAME = "signed_tail_attribute_sensitivity"
ANALYSIS_VERSION = 1
RANDOM_SEED = 20260909
DEFAULT_BOOTSTRAP_REPLICATES = 5000
SCREENS = ("primary", "conservative")
SIGNS = (
    ("T_plus", "T+", "positive_tail_count", "positive_tail_fraction", "positive_tail_multiple_of_normal"),
    ("T_minus", "T\u2212", "negative_tail_count", "negative_tail_fraction", "negative_tail_multiple_of_normal"),
)

# STOR_NID_2009 is preferred to STOR_NOR_2009 because the GAGES-II metadata
# explicitly warns that normal storage contains conspicuous zero values.
ATTRIBUTES = (
    "DRAIN_SQKM",
    "NDAMS_2009",
    "DDENS_2009",
    "MAJ_NDAMS_2009",
    "MAJ_DDENS_2009",
    "STOR_NID_2009",
    "CANALS_PCT",
    "PCT_IRRIG_AG",
    "FRESHW_WITHDRAWAL",
)

ATTRIBUTE_CATEGORIES = {
    "DRAIN_SQKM": "drainage_area",
    "NDAMS_2009": "dams",
    "DDENS_2009": "dams",
    "MAJ_NDAMS_2009": "dams",
    "MAJ_DDENS_2009": "dams",
    "STOR_NID_2009": "storage",
    "CANALS_PCT": "canals",
    "PCT_IRRIG_AG": "irrigation",
    "FRESHW_WITHDRAWAL": "withdrawal",
}

GAGES_FILES = {
    "conterm_basinid.txt": ["STAID", "DRAIN_SQKM"],
    "conterm_bas_classif.txt": [
        "STAID", "CLASS", "AGGECOREGION", "HYDRO_DISTURB_INDX"
    ],
    "conterm_hydromod_dams.txt": [
        "STAID",
        "NDAMS_2009",
        "DDENS_2009",
        "STOR_NID_2009",
        "STOR_NOR_2009",
        "MAJ_NDAMS_2009",
        "MAJ_DDENS_2009",
    ],
    "conterm_hydromod_other.txt": [
        "STAID",
        "CANALS_PCT",
        "CANALS_MAINSTEM_PCT",
        "FRESHW_WITHDRAWAL",
        "PCT_IRRIG_AG",
    ],
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tail-source",
        default=str(
            PROJECT_ROOT
            / "continental_run"
            / "sensitivity_analyses"
            / "tail_direction"
            / "basin_tail_direction_and_clustering.csv"
        ),
    )
    parser.add_argument(
        "--gages-archive",
        default=str(
            PROJECT_ROOT
            / "basinchar_and_report_sept_2011"
            / "spreadsheets-in-csv-format.zip"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "continental_run"
            / "sensitivity_analyses"
            / "signed_tail_attributes_v1"
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    with atomic_target(path) as temporary:
        frame.to_csv(temporary, index=False)


def write_text_atomic(text: str, path: Path) -> None:
    with atomic_target(path) as temporary:
        temporary.write_text(text, encoding="utf-8")


def stable_rng(*parts: object) -> np.random.Generator:
    label = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="little", signed=False)
    seed = (RANDOM_SEED + offset) % (2**63 - 1)
    return np.random.default_rng(seed)


def load_tail_source(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    receipt_path = path.parent / "_SUCCESS.json"
    if not receipt_path.exists():
        raise RuntimeError(f"Receipt is required for signed-tail input: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    recorded = receipt.get("output_sha256", {}).get(path.name)
    actual = sha256_file(path)
    if not recorded or recorded != actual:
        raise RuntimeError(
            f"Signed-tail input hash does not match its receipt: {recorded=} {actual=}"
        )
    if not receipt.get("metrics", {}).get("stage17_validation_passed", False):
        raise RuntimeError("Parent signed-tail receipt did not pass Stage-17 validation")

    tail = pd.read_csv(path, dtype={"GAGE_ID": str}, low_memory=False)
    expected_all_rows = receipt.get("metrics", {}).get("all_variant_rows")
    if expected_all_rows is not None and len(tail) != int(expected_all_rows):
        raise AssertionError(
            f"Parent tail row count differs from receipt: {len(tail)} != {expected_all_rows}"
        )
    tail["GAGE_ID"] = tail["GAGE_ID"].astype(str).str.zfill(8)
    tail = tail[tail["screen_variant"].isin(SCREENS)].copy()
    if tail.duplicated(["GAGE_ID", "screen_variant"]).any():
        raise AssertionError("Duplicate basin-screen rows in signed-tail input")
    return tail, receipt


def read_gages_member(archive: ZipFile, name: str, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(
        archive.open(name), encoding="latin-1", usecols=columns, dtype={"STAID": str}
    )
    frame["STAID"] = frame["STAID"].astype(str).str.zfill(8)
    if frame["STAID"].duplicated().any():
        raise AssertionError(f"Duplicate STAID values in {name}")
    return frame


def load_gages_attributes(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    with ZipFile(path) as archive:
        frames = [
            read_gages_member(archive, member, columns)
            for member, columns in GAGES_FILES.items()
        ]
        metadata = pd.read_csv(
            archive.open("variable_descriptions.txt"), encoding="latin-1", dtype=str
        )

    attributes = frames[0]
    for frame in frames[1:]:
        attributes = attributes.merge(frame, on="STAID", how="outer", validate="one_to_one")
    if len(attributes) != 9067:
        raise AssertionError(
            f"Expected 9,067 CONUS GAGES-II rows; found {len(attributes):,}"
        )

    numeric = [
        *ATTRIBUTES,
        "STOR_NOR_2009",
        "CANALS_MAINSTEM_PCT",
        "HYDRO_DISTURB_INDX",
    ]
    for column in numeric:
        attributes[column] = pd.to_numeric(attributes[column], errors="coerce")
        # The distributed text uses negative sentinels for some distance fields.
        # None are expected in the selected nonnegative fields, but fail closed.
        attributes.loc[attributes[column] < 0, column] = np.nan

    metadata = metadata[
        metadata["VARIABLE_NAME"].isin(
            [*ATTRIBUTES, "STOR_NOR_2009", "CLASS", "AGGECOREGION"]
        )
    ].copy()
    metadata = metadata.drop_duplicates("VARIABLE_NAME", keep="first")
    metadata["selected_for_association"] = metadata["VARIABLE_NAME"].isin(ATTRIBUTES)
    metadata["analysis_category"] = metadata["VARIABLE_NAME"].map(
        ATTRIBUTE_CATEGORIES
    )
    metadata["analysis_transform"] = np.where(
        metadata["selected_for_association"], "log1p then z-score", "not modeled"
    )
    metadata["analysis_note"] = ""
    metadata.loc[
        metadata["VARIABLE_NAME"].eq("STOR_NOR_2009"), "analysis_note"
    ] = (
        "Not modeled: GAGES-II metadata documents inconsistent zero normal-storage "
        "values; STOR_NID_2009 is used instead."
    )
    metadata.loc[
        metadata["VARIABLE_NAME"].eq("CLASS"), "analysis_note"
    ] = "Defines the official GAGES-II reference-basin clean subgroup."
    metadata.loc[
        metadata["VARIABLE_NAME"].eq("AGGECOREGION"), "analysis_note"
    ] = "Used as fixed effects and for stratified summaries."
    return attributes, metadata


def truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def merge_sources(tail: pd.DataFrame, attributes: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    gages = attributes.rename(
        columns={"STAID": "GAGE_ID", "AGGECOREGION": "GAGESII_AGGECOREGION"}
    )
    merged = tail.merge(
        gages, on="GAGE_ID", how="left", validate="many_to_one"
    )
    if merged["DRAIN_SQKM"].isna().any():
        missing = merged.loc[merged["DRAIN_SQKM"].isna(), "GAGE_ID"].unique()
        raise AssertionError(f"Unmatched GAGES-II attributes for {len(missing)} basin(s)")
    region_match = (
        merged["AGGECOREGION"].fillna("").astype(str)
        == merged["GAGESII_AGGECOREGION"].fillna("").astype(str)
    )
    if not region_match.all():
        raise AssertionError(f"{int((~region_match).sum())} ecoregion mismatches")
    reference = merged["CLASS"].astype(str).str.strip().str.lower().eq("ref")
    if "is_reference" in merged:
        reference_match = truthy(merged["is_reference"]).eq(reference)
        if not reference_match.all():
            raise AssertionError(
                f"{int((~reference_match).sum())} GAGES-II reference-flag mismatches"
            )
    merged["gagesii_reference"] = reference

    withdrawal_q25 = float(attributes["FRESHW_WITHDRAWAL"].quantile(0.25))
    merged["stringent_attribute_clean"] = (
        reference
        & merged["NDAMS_2009"].eq(0)
        & merged["STOR_NID_2009"].eq(0)
        & merged["MAJ_NDAMS_2009"].eq(0)
        & merged["CANALS_PCT"].eq(0)
        & merged["PCT_IRRIG_AG"].eq(0)
        & merged["FRESHW_WITHDRAWAL"].le(withdrawal_q25)
    )
    return merged, withdrawal_q25


def make_long_basin_table(merged: pd.DataFrame) -> pd.DataFrame:
    common = [
        "GAGE_ID",
        "screen_variant",
        "n_standardized",
        "normal_one_sided_tail_reference",
        "quality_tier",
        "AGGECOREGION",
        "CLASS",
        "gagesii_reference",
        "stringent_attribute_clean",
        "HYDRO_DISTURB_INDX",
        *ATTRIBUTES,
    ]
    parts = []
    for sign_code, sign_label, count_column, fraction_column, multiple_column in SIGNS:
        part = merged[common].copy()
        part["tail_sign"] = sign_code
        part["tail_sign_label"] = sign_label
        part["tail_count"] = merged[count_column].astype(int)
        part["raw_tail_fraction"] = merged[fraction_column].astype(float)
        part["raw_tail_multiple_of_normal"] = merged[multiple_column].astype(float)
        part["jeffreys_tail_fraction"] = (
            part["tail_count"] + 0.5
        ) / (part["n_standardized"] + 1.0)
        part["jeffreys_tail_multiple_of_normal"] = (
            part["jeffreys_tail_fraction"]
            / part["normal_one_sided_tail_reference"]
        )
        part["log_jeffreys_tail_multiple"] = np.log(
            part["jeffreys_tail_multiple_of_normal"]
        )
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    ordered = [
        "GAGE_ID",
        "screen_variant",
        "tail_sign",
        "tail_sign_label",
        "n_standardized",
        "tail_count",
        "raw_tail_fraction",
        "raw_tail_multiple_of_normal",
        "jeffreys_tail_fraction",
        "jeffreys_tail_multiple_of_normal",
        "log_jeffreys_tail_multiple",
        "normal_one_sided_tail_reference",
        "quality_tier",
        "AGGECOREGION",
        "CLASS",
        "gagesii_reference",
        "stringent_attribute_clean",
        "HYDRO_DISTURB_INDX",
        *ATTRIBUTES,
    ]
    return long[ordered].sort_values(
        ["screen_variant", "tail_sign", "GAGE_ID"]
    ).reset_index(drop=True)


def subgroup_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all_tail_estimable_basins": pd.Series(True, index=frame.index),
        "gagesii_reference": frame["gagesii_reference"].astype(bool),
        "stringent_attribute_clean": frame["stringent_attribute_clean"].astype(bool),
    }


def stratified_bootstrap_median(
    frame: pd.DataFrame,
    value_column: str,
    replicates: int,
    *seed_parts: object,
) -> tuple[float, float]:
    groups = [
        group[value_column].dropna().to_numpy(float)
        for _, group in frame.groupby("AGGECOREGION", dropna=False, sort=True)
    ]
    groups = [values for values in groups if len(values)]
    if not groups:
        return np.nan, np.nan
    rng = stable_rng("stratified_median", *seed_parts)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = np.concatenate(
            [rng.choice(values, size=len(values), replace=True) for values in groups]
        )
        estimates[index] = np.median(sample)
    return tuple(np.quantile(estimates, [0.025, 0.975]).astype(float))


def descriptive_row(group: pd.DataFrame) -> dict:
    normal = group["normal_one_sided_tail_reference"].to_numpy(float)
    if not np.allclose(normal, normal[0], rtol=0, atol=1e-15):
        raise AssertionError("Normal one-sided reference is not constant")
    total_n = int(group["n_standardized"].sum())
    total_events = int(group["tail_count"].sum())
    raw = group["raw_tail_multiple_of_normal"].to_numpy(float)
    smooth = group["jeffreys_tail_multiple_of_normal"].to_numpy(float)
    return {
        "basin_n": int(len(group)),
        "ecoregion_n": int(group["AGGECOREGION"].nunique(dropna=True)),
        "standardized_transition_n": total_n,
        "tail_event_n": total_events,
        "pooled_tail_fraction": total_events / total_n,
        "pooled_tail_multiple_of_normal": total_events / total_n / normal[0],
        "median_raw_tail_fraction": float(group["raw_tail_fraction"].median()),
        "q25_raw_tail_fraction": float(group["raw_tail_fraction"].quantile(0.25)),
        "q75_raw_tail_fraction": float(group["raw_tail_fraction"].quantile(0.75)),
        "median_raw_tail_multiple_of_normal": float(np.median(raw)),
        "q25_raw_tail_multiple_of_normal": float(np.quantile(raw, 0.25)),
        "q75_raw_tail_multiple_of_normal": float(np.quantile(raw, 0.75)),
        "median_jeffreys_tail_multiple_of_normal": float(np.median(smooth)),
        "raw_zero_tail_basin_n": int(np.sum(group["tail_count"].eq(0))),
        "basin_fraction_raw_tail_above_normal": float(np.mean(raw > 1.0)),
    }


def continental_summaries(long: pd.DataFrame, replicates: int) -> pd.DataFrame:
    rows = []
    for screen in SCREENS:
        for sign_code, sign_label, *_ in SIGNS:
            base = long[
                long["screen_variant"].eq(screen) & long["tail_sign"].eq(sign_code)
            ]
            for subgroup, mask in subgroup_masks(base).items():
                group = base[mask]
                row = {
                    "screen_variant": screen,
                    "subgroup": subgroup,
                    "tail_sign": sign_code,
                    "tail_sign_label": sign_label,
                    **descriptive_row(group),
                }
                low, high = stratified_bootstrap_median(
                    group,
                    "raw_tail_multiple_of_normal",
                    replicates,
                    screen,
                    subgroup,
                    sign_code,
                )
                row["stratified_bootstrap_median_ci_low"] = low
                row["stratified_bootstrap_median_ci_high"] = high
                row["bootstrap_replicates"] = replicates
                rows.append(row)
    return pd.DataFrame(rows)


def ecoregion_summaries(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for screen in SCREENS:
        for sign_code, sign_label, *_ in SIGNS:
            base = long[
                long["screen_variant"].eq(screen) & long["tail_sign"].eq(sign_code)
            ]
            for subgroup, mask in subgroup_masks(base).items():
                selected = base[mask]
                for ecoregion, group in selected.groupby(
                    "AGGECOREGION", dropna=False, sort=True
                ):
                    rows.append(
                        {
                            "screen_variant": screen,
                            "subgroup": subgroup,
                            "tail_sign": sign_code,
                            "tail_sign_label": sign_label,
                            "AGGECOREGION": ecoregion,
                            **descriptive_row(group),
                            "low_basin_support_lt20": bool(len(group) < 20),
                        }
                    )
    return pd.DataFrame(rows)


def analysis_populations(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all_tail_estimable_basins": pd.Series(True, index=frame.index),
        "gagesii_reference": frame["gagesii_reference"].astype(bool),
    }


def marginal_associations(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for screen in SCREENS:
        for sign_code, sign_label, *_ in SIGNS:
            base = long[
                long["screen_variant"].eq(screen) & long["tail_sign"].eq(sign_code)
            ]
            for population, mask in analysis_populations(base).items():
                selected = base[mask]
                for attribute in ATTRIBUTES:
                    pair = selected[
                        ["raw_tail_multiple_of_normal", attribute]
                    ].replace([np.inf, -np.inf], np.nan).dropna()
                    rho = p_value = np.nan
                    if (
                        len(pair) >= 30
                        and pair[attribute].nunique() >= 3
                        and pair["raw_tail_multiple_of_normal"].nunique() >= 3
                    ):
                        result = stats.spearmanr(
                            pair[attribute], pair["raw_tail_multiple_of_normal"]
                        )
                        rho, p_value = float(result.statistic), float(result.pvalue)
                    rows.append(
                        {
                            "screen_variant": screen,
                            "analysis_population": population,
                            "tail_sign": sign_code,
                            "tail_sign_label": sign_label,
                            "attribute": attribute,
                            "attribute_category": ATTRIBUTE_CATEGORIES[attribute],
                            "n": int(len(pair)),
                            "attribute_unique_n": int(pair[attribute].nunique()),
                            "attribute_zero_n": int(pair[attribute].eq(0).sum()),
                            "spearman_rho": rho,
                            "p_value": p_value,
                        }
                    )
    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = result.groupby(
        ["screen_variant", "analysis_population", "tail_sign"], sort=False
    )["p_value"].transform(lambda values: benjamini_hochberg(values.to_numpy(float)))
    return result


def design_matrix(
    frame: pd.DataFrame,
    attribute: str,
    binary_predictor: bool = False,
) -> tuple[pd.DataFrame, dict]:
    import statsmodels.api as sm

    transformed = frame[attribute].astype(float)
    transform_name = "identity"
    if not binary_predictor:
        transformed = np.log1p(transformed)
        transform_name = "log1p then z-score"
        sd = float(transformed.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"No predictor variation for {attribute}")
        predictor = (transformed - transformed.mean()) / sd
    else:
        sd = float(transformed.std(ddof=0))
        predictor = transformed
    design = pd.DataFrame({"attribute_effect": predictor}, index=frame.index)

    area_adjusted = attribute != "DRAIN_SQKM"
    if area_adjusted:
        log_area = np.log1p(frame["DRAIN_SQKM"].astype(float))
        area_sd = float(log_area.std(ddof=0))
        if not np.isfinite(area_sd) or area_sd <= 0:
            raise ValueError("No drainage-area variation")
        design["log_area_z"] = (log_area - log_area.mean()) / area_sd

    regions = pd.get_dummies(
        frame["AGGECOREGION"].fillna("Missing").astype(str),
        prefix="ecoregion",
        drop_first=True,
        dtype=float,
    )
    design = pd.concat([design, regions], axis=1)
    design = sm.add_constant(design.astype(float), has_constant="add")
    return design, {
        "predictor_transform": transform_name,
        "predictor_transformed_sd": sd,
        "drainage_area_adjusted": area_adjusted,
        "ecoregion_fixed_effect_n": int(frame["AGGECOREGION"].nunique()),
    }


def fit_hc3_model(
    frame: pd.DataFrame,
    response: str,
    attribute: str,
    binary_predictor: bool = False,
) -> dict:
    import statsmodels.api as sm

    needed = list(dict.fromkeys([response, attribute, "DRAIN_SQKM", "AGGECOREGION"]))
    work = frame[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < 30:
        raise ValueError(f"Only {len(work)} complete observations")
    design, details = design_matrix(work, attribute, binary_predictor=binary_predictor)
    fit = sm.OLS(work[response].astype(float), design).fit(cov_type="HC3")
    ci = fit.conf_int(alpha=0.05).loc["attribute_effect"]
    beta = float(fit.params["attribute_effect"])
    return {
        "n": int(fit.nobs),
        **details,
        "beta_log_tail_multiple": beta,
        "std_error_hc3": float(fit.bse["attribute_effect"]),
        "t_hc3": float(fit.tvalues["attribute_effect"]),
        "p_value": float(fit.pvalues["attribute_effect"]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "multiplicative_factor": float(math.exp(beta)),
        "multiplicative_factor_ci_low": float(math.exp(ci.iloc[0])),
        "multiplicative_factor_ci_high": float(math.exp(ci.iloc[1])),
        "r2": float(fit.rsquared),
        "adjusted_r2": float(fit.rsquared_adj),
    }


def fixed_effect_models(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for screen in SCREENS:
        for sign_code, sign_label, *_ in SIGNS:
            base = long[
                long["screen_variant"].eq(screen) & long["tail_sign"].eq(sign_code)
            ]
            for population, mask in analysis_populations(base).items():
                selected = base[mask]
                for attribute in ATTRIBUTES:
                    row = {
                        "screen_variant": screen,
                        "analysis_population": population,
                        "tail_sign": sign_code,
                        "tail_sign_label": sign_label,
                        "attribute": attribute,
                        "attribute_category": ATTRIBUTE_CATEGORIES[attribute],
                        "response": "log_jeffreys_tail_multiple",
                        "model": "OLS with HC3 SE, ecoregion fixed effects",
                    }
                    try:
                        row.update(
                            fit_hc3_model(
                                selected,
                                "log_jeffreys_tail_multiple",
                                attribute,
                            )
                        )
                        row["model_status"] = "ok"
                    except ValueError as error:
                        row.update({"model_status": "unavailable", "reason": str(error)})
                    rows.append(row)
    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = result.groupby(
        ["screen_variant", "analysis_population", "tail_sign"], sort=False
    )["p_value"].transform(lambda values: benjamini_hochberg(values.to_numpy(float)))
    return result


def signed_contrast_models(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for screen in SCREENS:
        base = merged[merged["screen_variant"].eq(screen)].copy()
        for population, mask in analysis_populations(base).items():
            selected = base[mask]
            for attribute in ATTRIBUTES:
                row = {
                    "screen_variant": screen,
                    "analysis_population": population,
                    "attribute": attribute,
                    "attribute_category": ATTRIBUTE_CATEGORIES[attribute],
                    "response": "log[(N_plus+0.5)/(N_minus+0.5)]",
                    "positive_beta_means": "the T+ slope is more positive than the T\u2212 slope",
                    "model": "OLS with HC3 SE, ecoregion fixed effects",
                }
                try:
                    fit = fit_hc3_model(
                        selected, "jeffreys_tail_log_count_ratio", attribute
                    )
                    row.update(fit)
                    row["model_status"] = "ok"
                except ValueError as error:
                    row.update({"model_status": "unavailable", "reason": str(error)})
                rows.append(row)
    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = result.groupby(
        ["screen_variant", "analysis_population"], sort=False
    )["p_value"].transform(lambda values: benjamini_hochberg(values.to_numpy(float)))
    return result


def paired_bootstrap_ci(values: np.ndarray, replicates: int, *parts: object) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = stable_rng("paired_median", *parts)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        estimates[index] = np.median(rng.choice(values, size=len(values), replace=True))
    return tuple(np.quantile(estimates, [0.025, 0.975]).astype(float))


def conservative_screen_pairs(long: pd.DataFrame, replicates: int) -> pd.DataFrame:
    rows = []
    for sign_code, sign_label, *_ in SIGNS:
        sign = long[long["tail_sign"].eq(sign_code)].copy()
        for subgroup, mask in subgroup_masks(sign).items():
            selected = sign[mask]
            pivot = selected.pivot(
                index="GAGE_ID",
                columns="screen_variant",
                values="log_jeffreys_tail_multiple",
            ).dropna(subset=["primary", "conservative"])
            diff = (pivot["conservative"] - pivot["primary"]).to_numpy(float)
            rho = stats.spearmanr(pivot["primary"], pivot["conservative"])
            nonzero = diff[~np.isclose(diff, 0.0, rtol=0.0, atol=0.0)]
            wilcoxon_p = (
                float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else np.nan
            )
            low, high = paired_bootstrap_ci(
                diff, replicates, sign_code, subgroup, "conservative_minus_primary"
            )
            rows.append(
                {
                    "subgroup": subgroup,
                    "tail_sign": sign_code,
                    "tail_sign_label": sign_label,
                    "paired_basin_n": int(len(pivot)),
                    "spearman_primary_vs_conservative": float(rho.statistic),
                    "spearman_p_value": float(rho.pvalue),
                    "median_log_conservative_to_primary_ratio": float(np.median(diff)),
                    "median_conservative_to_primary_ratio": float(np.exp(np.median(diff))),
                    "paired_bootstrap_log_ratio_ci_low": low,
                    "paired_bootstrap_log_ratio_ci_high": high,
                    "wilcoxon_signed_rank_p_value": wilcoxon_p,
                    "fraction_conservative_above_primary": float(np.mean(diff > 0)),
                    "bootstrap_replicates": replicates,
                }
            )
    return pd.DataFrame(rows)


def clean_contrasts(long: pd.DataFrame) -> pd.DataFrame:
    definitions = (
        ("gagesii_reference", "gagesii_reference"),
        ("stringent_attribute_clean", "stringent_attribute_clean"),
    )
    rows = []
    for screen in SCREENS:
        for sign_code, sign_label, *_ in SIGNS:
            base = long[
                long["screen_variant"].eq(screen) & long["tail_sign"].eq(sign_code)
            ].copy()
            for definition, flag in definitions:
                clean = base[flag].astype(bool)
                row = {
                    "screen_variant": screen,
                    "tail_sign": sign_code,
                    "tail_sign_label": sign_label,
                    "clean_definition": definition,
                    "clean_basin_n": int(clean.sum()),
                    "comparison_basin_n": int((~clean).sum()),
                    "clean_median_raw_tail_multiple": float(
                        base.loc[clean, "raw_tail_multiple_of_normal"].median()
                    ),
                    "comparison_median_raw_tail_multiple": float(
                        base.loc[~clean, "raw_tail_multiple_of_normal"].median()
                    ),
                    "model": "clean indicator + log-area + ecoregion fixed effects; HC3 SE",
                }
                model_frame = base.assign(clean_indicator=clean.astype(float))
                try:
                    fit = fit_hc3_model(
                        model_frame,
                        "log_jeffreys_tail_multiple",
                        "clean_indicator",
                        binary_predictor=True,
                    )
                    row.update(fit)
                    row["model_status"] = "ok"
                except ValueError as error:
                    row.update({"model_status": "unavailable", "reason": str(error)})
                rows.append(row)
    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = result.groupby(
        ["screen_variant", "tail_sign"], sort=False
    )["p_value"].transform(lambda values: benjamini_hochberg(values.to_numpy(float)))
    return result


def attribute_coverage(merged: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    primary = merged[merged["screen_variant"].eq("primary")]
    metadata_by_name = metadata.set_index("VARIABLE_NAME")
    rows = []
    for attribute in ATTRIBUTES:
        values = primary[attribute]
        source = metadata_by_name.loc[attribute]
        rows.append(
            {
                "attribute": attribute,
                "attribute_category": ATTRIBUTE_CATEGORIES[attribute],
                "units": source.get("UNITS (numeric values)"),
                "primary_basin_n": int(len(values)),
                "nonmissing_n": int(values.notna().sum()),
                "missing_n": int(values.isna().sum()),
                "zero_n": int(values.eq(0).sum()),
                "minimum": float(values.min()),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
                "maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def clean_definitions(withdrawal_q25: float, merged: pd.DataFrame) -> pd.DataFrame:
    primary = merged[merged["screen_variant"].eq("primary")]
    conservative = merged[merged["screen_variant"].eq("conservative")]
    return pd.DataFrame(
        [
            {
                "clean_definition": "gagesii_reference",
                "criteria": "GAGES-II CLASS equals Ref (least-disturbed hydrologic condition)",
                "primary_basin_n": int(primary["gagesii_reference"].sum()),
                "conservative_basin_n": int(conservative["gagesii_reference"].sum()),
                "limitation": "Reference means least disturbed, not pristine or unregulated.",
            },
            {
                "clean_definition": "stringent_attribute_clean",
                "criteria": (
                    "GAGES-II reference AND reported NDAMS_2009=0, STOR_NID_2009=0, "
                    "MAJ_NDAMS_2009=0, CANALS_PCT=0, PCT_IRRIG_AG=0, and "
                    f"FRESHW_WITHDRAWAL <= CONUS GAGES-II q25 ({withdrawal_q25:g})"
                ),
                "primary_basin_n": int(primary["stringent_attribute_clean"].sum()),
                "conservative_basin_n": int(
                    conservative["stringent_attribute_clean"].sum()
                ),
                "limitation": (
                    "Small and regionally imbalanced; zeros refer to values rounded in the "
                    "distributed GAGES-II text release."
                ),
            },
        ]
    )


def methods_text(withdrawal_q25: float, replicates: int) -> str:
    return f"""# Signed-tail attribute sensitivity, version 1

This isolated sidecar reads the receipt-verified basin output from
`sensitivity_tail_direction_clustering.py` and the local official GAGES-II
CONUS text archive. It does not read daily records, refit innovations, or alter
the numbered workflow stages.

## Outcomes

Positive and negative extremes are kept separate. The reported raw basin tail
multiples are `T+ = [N(z > 3)/N] / Phi(-3)` and
`T- = [N(z < -3)/N] / Phi(-3)`. Continental and ecoregion tables preserve raw
basin medians, interquartile ranges, pooled counts, and the Gaussian reference.
For logarithmic regressions only, a Jeffreys half-count gives
`T_J = [(N_tail + 0.5)/(N + 1)] / Phi(-3)`. This prevents undefined logs for
basins with zero signed extremes; it does not replace the raw reported
estimand. Median intervals use {replicates:,} basin resamples stratified by the
nine aggregated ecoregions with deterministic seed {RANDOM_SEED}.

## Basin attributes and models

Nine prespecified GAGES-II variables cover drainage area, dam count and density,
major-dam count and density, NID storage per area, canal coverage, irrigated
agriculture, and freshwater withdrawal. Each continuous/nonnegative predictor
is transformed as `log(1+x)` and standardized. Each attribute is fitted in its
own equal-basin OLS model for `log(T_J)`, with aggregated-ecoregion fixed effects,
HC3 heteroscedasticity-robust standard errors, and log drainage area as a
covariate except when area itself is the predictor. Separate models are stored
for T+ and T-, for the primary and conservative innovation screens, and for all
tail-estimable basins and GAGES-II reference basins. Benjamini-Hochberg adjustment is applied
within each screen-by-population-by-sign family of nine tests. Marginal
Spearman associations are also retained.

The signed contrast models use the paired response
`log[(N_plus+0.5)/(N_minus+0.5)]`; a positive coefficient means the T+ slope is
more positive than the T- slope. It does not necessarily have the larger
absolute magnitude. These contrasts are descriptive tests of differential
association, not attribution of a physical cause.

The conservative screen is the archived variant that excludes estimated-value
endpoints, the lowest 1% of positive endpoint flows, and exact-zero increments,
with the variance law and standardization refitted. Its agreement with the
primary screen is summarized only among paired basins.

## Clean subgroups

The primary clean subgroup is the official GAGES-II reference class, defined
by GAGES-II as least-disturbed hydrologic condition. A stringent sensitivity
also requires reported zero dams, NID storage, major dams, canal coverage, and
irrigated agriculture, plus freshwater withdrawal no greater than the 25th
percentile across all 9,067 CONUS GAGES-II basins ({withdrawal_q25:g} in the
source units). The stringent subset is deliberately reported separately because
it is small and regionally imbalanced. Zero values are those in the distributed
rounded GAGES-II text archive.

## Boundaries

This is an observational sensitivity, not a causal model. Predictor z-scores
are computed separately inside each screen and population, so coefficient
magnitudes should not be compared as though one common standard deviation were
used. The equal-basin regressions model Jeffreys-smoothed empirical tail rates,
not latent event-level probabilities; record lengths vary, first-stage tail-rate
uncertainty is not propagated, and negative-tail counts include many zeros.
The GAGES-II
attributes are heterogeneous historical snapshots (freshwater withdrawal is
1995--2000, irrigation is 2002, dams are 2009, and the canal metadata do not
state a period), whereas the discharge analysis spans 1980--2025. Basin aggregates do not
encode operating schedules, water transfers, unrecorded structures, or the
timing of withdrawals. The models and basin resamples do not eliminate nested-
gage or within-ecoregion spatial dependence. HC3 intervals, nominal p-values,
and within-family BH discoveries are therefore exploratory rather than fully
calibrated for spatial dependence. BH adjustment is separate for each stated
nine-test family; discovery counts summed across families have no single 5%
FDR guarantee, and paired-screen p-values are unadjusted. The one-attribute-at-
a-time models also do not resolve multi-attribute collinearity. A T+ association therefore cannot
identify unscreened hydrologic inputs, regulation, or observation corrections,
and persistence of T- in a clean subset cannot prove a natural mechanism.
`STOR_NOR_2009` is not modeled because its GAGES-II metadata documents
inconsistent zero normal-storage values; `STOR_NID_2009` is used instead.
"""


def validation_checks(
    tail: pd.DataFrame,
    merged: pd.DataFrame,
    long: pd.DataFrame,
    attributes: pd.DataFrame,
    withdrawal_q25: float,
    parent_receipt: dict,
) -> dict:
    raw_positive = long[long["tail_sign"].eq("T_plus")].sort_values(
        ["screen_variant", "GAGE_ID"]
    )["raw_tail_multiple_of_normal"].to_numpy(float)
    expected_positive = merged.sort_values(["screen_variant", "GAGE_ID"])[
        "positive_tail_multiple_of_normal"
    ].to_numpy(float)
    raw_negative = long[long["tail_sign"].eq("T_minus")].sort_values(
        ["screen_variant", "GAGE_ID"]
    )["raw_tail_multiple_of_normal"].to_numpy(float)
    expected_negative = merged.sort_values(["screen_variant", "GAGE_ID"])[
        "negative_tail_multiple_of_normal"
    ].to_numpy(float)
    strict = merged["stringent_attribute_clean"]
    strict_valid = (
        merged.loc[strict, "gagesii_reference"].all()
        and merged.loc[strict, "NDAMS_2009"].eq(0).all()
        and merged.loc[strict, "STOR_NID_2009"].eq(0).all()
        and merged.loc[strict, "MAJ_NDAMS_2009"].eq(0).all()
        and merged.loc[strict, "CANALS_PCT"].eq(0).all()
        and merged.loc[strict, "PCT_IRRIG_AG"].eq(0).all()
        and merged.loc[strict, "FRESHW_WITHDRAWAL"].le(withdrawal_q25).all()
    )
    checks = {
        "parent_tail_stage17_validation_passed": bool(
            parent_receipt.get("metrics", {}).get("stage17_validation_passed", False)
        ),
        "selected_tail_screens_exact": bool(
            set(tail["screen_variant"].unique()) == set(SCREENS)
        ),
        "primary_tail_rows_match_parent_receipt": bool(
            int(tail["screen_variant"].eq("primary").sum())
            == int(parent_receipt.get("metrics", {}).get("primary_basins_analyzed", -1))
        ),
        "all_tail_rows_match_gagesii": bool(merged["DRAIN_SQKM"].notna().all()),
        "gagesii_universe_is_9067": bool(len(attributes) == 9067),
        "one_row_per_basin_screen": bool(
            not merged.duplicated(["GAGE_ID", "screen_variant"]).any()
        ),
        "long_has_two_rows_per_basin_screen": bool(len(long) == 2 * len(merged)),
        "long_has_both_signed_outcomes": bool(
            set(long["tail_sign"]) == {"T_plus", "T_minus"}
        ),
        "positive_raw_tail_values_preserved": bool(
            np.allclose(raw_positive, expected_positive, rtol=0, atol=1e-14)
        ),
        "negative_raw_tail_values_preserved": bool(
            np.allclose(raw_negative, expected_negative, rtol=0, atol=1e-14)
        ),
        "jeffreys_outcomes_positive_finite": bool(
            np.isfinite(long["jeffreys_tail_multiple_of_normal"]).all()
            and long["jeffreys_tail_multiple_of_normal"].gt(0).all()
        ),
        "raw_tail_fraction_matches_counts": bool(
            np.allclose(
                long["raw_tail_fraction"],
                long["tail_count"] / long["n_standardized"],
                rtol=0,
                atol=1e-14,
            )
        ),
        "raw_tail_multiple_matches_fraction": bool(
            np.allclose(
                long["raw_tail_multiple_of_normal"],
                long["raw_tail_fraction"]
                / long["normal_one_sided_tail_reference"],
                rtol=0,
                atol=2e-12,
            )
        ),
        "selected_attributes_complete": bool(
            merged[list(ATTRIBUTES)].notna().all().all()
        ),
        "selected_attributes_nonnegative": bool(
            (merged[list(ATTRIBUTES)].dropna() >= 0).all().all()
        ),
        "stringent_clean_definition_holds": bool(strict_valid),
        "stringent_clean_has_primary_support": bool(
            merged.loc[merged["screen_variant"].eq("primary"), "stringent_attribute_clean"].sum()
            >= 30
        ),
        "stringent_clean_has_conservative_support": bool(
            merged.loc[
                merged["screen_variant"].eq("conservative"),
                "stringent_attribute_clean",
            ].sum()
            >= 30
        ),
    }
    return {
        "checks": checks,
        "passed_n": int(sum(checks.values())),
        "check_n": int(len(checks)),
        "all_checks_passed": bool(all(checks.values())),
    }


def build_analysis_summary(
    continental: pd.DataFrame,
    models: pd.DataFrame,
    signed_models: pd.DataFrame,
    paired: pd.DataFrame,
    validation: dict,
) -> dict:
    primary = continental[
        continental["screen_variant"].eq("primary")
        & continental["subgroup"].eq("all_tail_estimable_basins")
    ]
    conservative = continental[
        continental["screen_variant"].eq("conservative")
        & continental["subgroup"].eq("all_tail_estimable_basins")
    ]

    def median_map(frame: pd.DataFrame) -> dict:
        return {
            row.tail_sign: {
                "basin_n": int(row.basin_n),
                "median_raw_tail_multiple": float(
                    row.median_raw_tail_multiple_of_normal
                ),
                "stratified_bootstrap_ci": [
                    float(row.stratified_bootstrap_median_ci_low),
                    float(row.stratified_bootstrap_median_ci_high),
                ],
            }
            for row in frame.itertuples(index=False)
        }

    significant = models[
        models["model_status"].eq("ok") & models["p_fdr_bh"].lt(0.05)
    ]
    signed_significant = signed_models[
        signed_models["model_status"].eq("ok")
        & signed_models["p_fdr_bh"].lt(0.05)
    ]
    all_pairs = paired[paired["subgroup"].eq("all_tail_estimable_basins")]
    return {
        "analysis": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "primary_all_basin_signed_medians": median_map(primary),
        "conservative_all_basin_signed_medians": median_map(conservative),
        "sum_of_within_family_fdr_discovery_rows_signed_models": int(
            len(significant)
        ),
        "sum_of_within_family_fdr_discovery_rows_differential_sign_models": int(
            len(signed_significant)
        ),
        "multiple_testing_note": (
            "Counts sum discoveries across separately adjusted families and do not "
            "carry one global 5% FDR guarantee."
        ),
        "all_basin_primary_conservative_pairs": {
            row.tail_sign: {
                "paired_basin_n": int(row.paired_basin_n),
                "spearman": float(row.spearman_primary_vs_conservative),
                "median_conservative_to_primary_ratio": float(
                    row.median_conservative_to_primary_ratio
                ),
            }
            for row in all_pairs.itertuples(index=False)
        },
        "validation": validation,
    }


def main() -> None:
    args = parse_arguments()
    if args.bootstrap_replicates < 1000:
        raise ValueError("At least 1,000 bootstrap replicates are required")
    tail_path = Path(args.tail_source).expanduser().resolve()
    gages_path = Path(args.gages_archive).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    receipt_path = out / "_SUCCESS.json"
    if receipt_path.exists() and not args.force:
        raise RuntimeError(f"Outputs already complete at {out}; use --force to replace them")
    out.mkdir(parents=True, exist_ok=True)
    if args.force:
        receipt_path.unlink(missing_ok=True)

    tail, parent_receipt = load_tail_source(tail_path)
    gages, metadata = load_gages_attributes(gages_path)
    merged, withdrawal_q25 = merge_sources(tail, gages)
    long = make_long_basin_table(merged)

    continental = continental_summaries(long, args.bootstrap_replicates)
    ecoregion = ecoregion_summaries(long)
    marginal = marginal_associations(long)
    models = fixed_effect_models(long)
    signed_models = signed_contrast_models(merged)
    paired = conservative_screen_pairs(long, args.bootstrap_replicates)
    clean = clean_contrasts(long)
    coverage = attribute_coverage(merged, metadata)
    definitions = clean_definitions(withdrawal_q25, merged)
    validation = validation_checks(
        tail, merged, long, gages, withdrawal_q25, parent_receipt
    )
    if not validation["all_checks_passed"]:
        failed = [
            name for name, passed in validation["checks"].items() if not passed
        ]
        raise AssertionError(f"Validation failed: {failed}")

    summary = build_analysis_summary(
        continental, models, signed_models, paired, validation
    )
    paths = {
        "basin": out / "basin_signed_tail_attribute_analysis.csv",
        "continental": out / "continental_signed_tail_summary.csv",
        "ecoregion": out / "ecoregion_signed_tail_summary.csv",
        "marginal": out / "all_basin_attribute_associations.csv",
        "models": out / "ecoregion_fixed_effect_attribute_models.csv",
        "signed_models": out / "signed_tail_attribute_contrasts.csv",
        "paired": out / "conservative_screen_paired_summary.csv",
        "clean": out / "clean_subgroup_contrasts.csv",
        "definitions": out / "clean_subgroup_definitions.csv",
        "coverage": out / "attribute_coverage.csv",
        "metadata": out / "attribute_metadata.csv",
        "methods": out / "METHODS.md",
        "summary": out / "ANALYSIS_SUMMARY.json",
    }
    write_csv_atomic(long, paths["basin"])
    write_csv_atomic(continental, paths["continental"])
    write_csv_atomic(ecoregion, paths["ecoregion"])
    write_csv_atomic(marginal, paths["marginal"])
    write_csv_atomic(models, paths["models"])
    write_csv_atomic(signed_models, paths["signed_models"])
    write_csv_atomic(paired, paths["paired"])
    write_csv_atomic(clean, paths["clean"])
    write_csv_atomic(definitions, paths["definitions"])
    write_csv_atomic(coverage, paths["coverage"])
    write_csv_atomic(metadata, paths["metadata"])
    write_text_atomic(
        methods_text(withdrawal_q25, args.bootstrap_replicates), paths["methods"]
    )
    atomic_json(paths["summary"], summary)

    script_path = Path(__file__).resolve()
    common_path = PROJECT_ROOT / "scripts" / "lib" / "common.py"
    statistics_path = PROJECT_ROOT / "scripts" / "lib" / "statistics.py"
    output_hashes = {path.name: sha256_file(path) for path in paths.values()}
    receipt = {
        "analysis": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "completed_utc": utc_now(),
        "code_sha256": {
            str(script_path.relative_to(PROJECT_ROOT)): sha256_file(script_path),
            str(common_path.relative_to(PROJECT_ROOT)): sha256_file(common_path),
            str(statistics_path.relative_to(PROJECT_ROOT)): sha256_file(statistics_path),
        },
        "input_sha256": {
            str(tail_path.relative_to(PROJECT_ROOT)): sha256_file(tail_path),
            str((tail_path.parent / "_SUCCESS.json").relative_to(PROJECT_ROOT)): sha256_file(
                tail_path.parent / "_SUCCESS.json"
            ),
            str(gages_path.relative_to(PROJECT_ROOT)): sha256_file(gages_path),
        },
        "parent_tail_recorded_sha256": parent_receipt["output_sha256"][tail_path.name],
        "parameters": {
            "screens": list(SCREENS),
            "attributes": list(ATTRIBUTES),
            "bootstrap_replicates": args.bootstrap_replicates,
            "random_seed": RANDOM_SEED,
            "stringent_clean_withdrawal_q25_conus": withdrawal_q25,
        },
        "metrics": {
            "gagesii_conus_basin_n": int(len(gages)),
            "tail_basin_screen_row_n": int(len(merged)),
            "signed_basin_row_n": int(len(long)),
            "primary_basin_n": int(merged["screen_variant"].eq("primary").sum()),
            "conservative_basin_n": int(
                merged["screen_variant"].eq("conservative").sum()
            ),
            "primary_reference_basin_n": int(
                merged.loc[
                    merged["screen_variant"].eq("primary"), "gagesii_reference"
                ].sum()
            ),
            "conservative_reference_basin_n": int(
                merged.loc[
                    merged["screen_variant"].eq("conservative"),
                    "gagesii_reference",
                ].sum()
            ),
            "primary_stringent_clean_basin_n": int(
                merged.loc[
                    merged["screen_variant"].eq("primary"),
                    "stringent_attribute_clean",
                ].sum()
            ),
            "conservative_stringent_clean_basin_n": int(
                merged.loc[
                    merged["screen_variant"].eq("conservative"),
                    "stringent_attribute_clean",
                ].sum()
            ),
            "validation_passed_n": validation["passed_n"],
            "validation_check_n": validation["check_n"],
        },
        "validation": validation,
        "output_sha256": output_hashes,
        "outputs": [str(path) for path in paths.values()],
        "python": sys.version.split()[0],
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "platform": platform.platform(),
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {len(paths) + 1} files to {out}")


if __name__ == "__main__":
    main()
