#!/usr/bin/env python3
"""Reconstruct the prespecified regional-medoid example-gage selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_MODEL = "regularized_power_state_empirical"
REGION_ORDER = (
    "WestXeric",
    "WestMnts",
    "WestPlains",
    "CntlPlains",
    "MxWdShld",
    "SEPlains",
    "SECstPlain",
    "EastHghlnds",
    "NorthEast",
)
REGION_LABELS = {
    "WestXeric": "Western Xeric",
    "WestMnts": "Western Mountains",
    "WestPlains": "Western Plains",
    "CntlPlains": "Central Plains",
    "MxWdShld": "Mixed Wood Shield",
    "SEPlains": "Southeast Plains",
    "SECstPlain": "Southeast Coastal Plain",
    "EastHghlnds": "Eastern Highlands",
    "NorthEast": "Northeast",
}
EXPECTED_GAGES = {
    "WestXeric": "09394500",
    "WestMnts": "12431000",
    "WestPlains": "07328500",
    "CntlPlains": "05584500",
    "MxWdShld": "05129115",
    "SEPlains": "02039500",
    "SECstPlain": "02092500",
    "EastHghlnds": "02142000",
    "NorthEast": "01381500",
}
DISTANCE_COORDINATES = (
    "variance_exponent",
    "variance_r2",
    "standardized_tail_fraction",
    "standardized_excess_kurtosis",
    "n_transitions",
    "maximum_pairwise_flow_group_ks",
)


def normalize_gage(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)


def select_regional_medoids(
    distribution: pd.DataFrame, prediction: pd.DataFrame
) -> pd.DataFrame:
    """Select one central, well-supported example within each ecoregion."""
    distribution = distribution.copy()
    prediction = prediction.copy()
    distribution["GAGE_ID"] = normalize_gage(distribution["GAGE_ID"])
    prediction["GAGE_ID"] = normalize_gage(prediction["GAGE_ID"])
    prediction_available = set(
        prediction.loc[
            prediction["lead_days"].eq(3)
            & prediction["threshold_name"].eq("Q10")
            & prediction["model"].eq(MAIN_MODEL),
            "GAGE_ID",
        ]
    )
    eligible = distribution.loc[
        distribution["quality_tier"].eq("TIER1_HIGH_QUALITY")
        & pd.to_numeric(distribution["n_transitions"], errors="coerce").ge(1_000)
        & pd.to_numeric(distribution["variance_exponent"], errors="coerce").notna()
        & distribution["GAGE_ID"].isin(prediction_available)
    ].copy()

    rows: list[pd.Series] = []
    for region in REGION_ORDER:
        candidates = eligible[eligible["AGGECOREGION"].eq(region)].copy()
        if candidates.empty:
            raise RuntimeError(f"No eligible example gage in {region}")
        transformed = candidates[list(DISTANCE_COORDINATES)].apply(
            pd.to_numeric, errors="coerce"
        )
        transformed["standardized_excess_kurtosis"] = np.log1p(
            transformed["standardized_excess_kurtosis"].clip(lower=0)
        )
        transformed["n_transitions"] = np.log(transformed["n_transitions"])
        center = transformed.median()
        mad = (transformed - center).abs().median().replace(0, np.nan)
        robust_coordinates = (transformed - center).div(mad).fillna(0)
        candidates["medoid_distance"] = np.sqrt(
            np.square(robust_coordinates).sum(axis=1)
        )
        chosen = candidates.sort_values(["medoid_distance", "GAGE_ID"]).iloc[0].copy()
        chosen["regional_candidate_pool"] = int(len(candidates))
        rows.append(chosen)

    selected = pd.DataFrame(rows)
    selected["region_label"] = selected["AGGECOREGION"].map(REGION_LABELS)
    selected["tail_multiple"] = (
        selected["standardized_tail_fraction"]
        / selected["normal_tail_reference"]
    )
    selected["display_order"] = selected["AGGECOREGION"].map(
        {region: index + 1 for index, region in enumerate(REGION_ORDER)}
    )
    selected = selected.sort_values("display_order").reset_index(drop=True)

    actual = dict(zip(selected["AGGECOREGION"], selected["GAGE_ID"]))
    if actual != EXPECTED_GAGES:
        raise RuntimeError(f"Regional-medoid selection changed: {actual}")
    columns = [
        "display_order",
        "AGGECOREGION",
        "region_label",
        "GAGE_ID",
        "regional_candidate_pool",
        "medoid_distance",
        "n_transitions",
        "variance_exponent",
        "variance_r2",
        "tail_multiple",
        "standardized_excess_kurtosis",
        "maximum_pairwise_flow_group_ks",
    ]
    return selected[columns]


def select_from_release() -> pd.DataFrame:
    derived = REPO_ROOT / "data" / "derived"
    distribution = pd.read_csv(
        derived / "basin_distribution.csv.gz", dtype={"GAGE_ID": "string"}
    )
    prediction = pd.read_csv(
        derived / "basin_prediction_q10.csv.gz", dtype={"GAGE_ID": "string"}
    )
    return select_regional_medoids(distribution, prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "example" / "example_gage_selection.csv",
    )
    args = parser.parse_args()
    selected = select_from_release()
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(destination, index=False)
    print(selected.to_string(index=False))
    try:
        shown = destination.relative_to(REPO_ROOT)
    except ValueError:
        shown = destination
    print(f"\nWrote {shown}")


if __name__ == "__main__":
    main()
