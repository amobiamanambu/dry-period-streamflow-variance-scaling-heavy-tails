#!/usr/bin/env python3
"""Recompute the article's registered numerical claims from frozen products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

HUC02_NAMES = {
    "01": "New England",
    "02": "Mid Atlantic",
    "03": "South Atlantic-Gulf",
    "04": "Great Lakes",
    "05": "Ohio",
    "06": "Tennessee",
    "07": "Upper Mississippi",
    "08": "Lower Mississippi",
    "09": "Souris-Red-Rainy",
    "10L": "Lower Missouri",
    "10U": "Upper Missouri",
    "11": "Arkansas-White-Red",
    "12": "Texas-Gulf",
    "13": "Rio Grande",
    "14": "Upper Colorado",
    "15": "Lower Colorado",
    "16": "Great Basin",
    "17": "Pacific Northwest",
    "18": "California",
}

PLACE_METRICS = (
    "lambda_star",
    "positive_tail_multiple_of_normal",
    "negative_tail_multiple_of_normal",
    "gamma_direct",
    "weighted_negative_within_share",
    "relative_crps_gain_percent",
)


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def random_effects(frame: pd.DataFrame) -> dict[str, float]:
    work = frame[[
        "recession_variance_exponent", "recession_variance_exponent_ci_low",
        "recession_variance_exponent_ci_high",
    ]].apply(pd.to_numeric, errors="coerce").dropna()
    work["se"] = (
        work["recession_variance_exponent_ci_high"]
        - work["recession_variance_exponent_ci_low"]
    ) / (2.0 * 1.96)
    work = work[work["se"].gt(0)]
    y = work["recession_variance_exponent"].to_numpy(float)
    variance = work["se"].to_numpy(float) ** 2
    weight = 1.0 / variance
    fixed = float(np.sum(weight * y) / np.sum(weight))
    q = float(np.sum(weight * (y - fixed) ** 2))
    degrees = len(y) - 1
    denominator = float(np.sum(weight) - np.sum(weight**2) / np.sum(weight))
    tau2 = max(0.0, (q - degrees) / denominator)
    random_weight = 1.0 / (variance + tau2)
    center = float(np.sum(random_weight * y) / np.sum(random_weight))
    center_se = float(np.sqrt(1.0 / np.sum(random_weight)))
    prediction_se = float(np.sqrt(tau2 + center_se**2))
    return {
        "m_random_effects_center": center,
        "m_prediction_low": center - 1.96 * prediction_se,
        "m_prediction_high": center + 1.96 * prediction_se,
        "m_i2_percent": max(0.0, (q - degrees) / q * 100.0),
    }


def relative_crps_skill(
    prediction: pd.DataFrame, model: str, reference: str, lead: int
) -> float:
    keys = ["GAGE_ID", "spatial_group", "lead_days"]
    selected = prediction[prediction["lead_days"].eq(lead)]
    baseline = selected[selected["model"].eq(reference)][keys + ["n", "crps"]]
    alternative = selected[selected["model"].eq(model)][keys + ["n", "crps"]]
    paired = baseline.merge(
        alternative, on=keys, suffixes=("_reference", "_model"), validate="one_to_one"
    ).dropna(subset=["crps_reference", "crps_model", "n_reference"])
    weights = paired["n_reference"].to_numpy(float)
    improvement = paired["crps_reference"].to_numpy(float) - paired["crps_model"].to_numpy(float)
    reference_score = float(np.average(paired["crps_reference"], weights=weights))
    return float(np.average(improvement, weights=weights) / reference_score)


def bootstrap_supported(
    summary: pd.DataFrame,
    model: str,
    reference: str,
    lead: int,
    minimum_basins: int,
    minimum_spatial_groups: int,
) -> float:
    """Return the prespecified support decision for one Q10 CRPS contrast."""
    selected = summary[
        summary["threshold_name"].eq("Q10")
        & summary["metric"].eq("crps")
        & summary["model"].eq(model)
        & summary["reference_model"].eq(reference)
        & summary["lead_days"].eq(lead)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one Q10 CRPS summary row for {model} versus {reference} "
            f"at {lead} day(s); found {len(selected)}"
        )
    row = selected.iloc[0]
    supported = (
        int(row["basins"]) >= minimum_basins
        and int(row["spatial_groups"]) >= minimum_spatial_groups
        and float(row["bootstrap_ci_low"]) > 0.0
    )
    return float(supported)


def summarize_huc02(place: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the descriptive HUC2 summaries used for place comparisons."""
    rows: list[dict[str, float | str]] = []
    for code, name in HUC02_NAMES.items():
        group = place[place["HUC02"].eq(code)]
        row: dict[str, float | str] = {"HUC02": code, "region": name}
        for metric in PLACE_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_n"] = float(len(values))
            row[f"{metric}_q25"] = float(values.quantile(0.25))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_q75"] = float(values.quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_claims() -> dict[str, float]:
    derived = REPO_ROOT / "data" / "derived"
    sample = pd.read_csv(derived / "basin_sample.csv.gz", dtype={"GAGE_ID": "string"})
    scaling = pd.read_csv(derived / "basin_scaling.csv.gz", dtype={"GAGE_ID": "string"})
    distribution = pd.read_csv(
        derived / "basin_distribution.csv.gz", dtype={"GAGE_ID": "string"}
    )
    signed = pd.read_csv(derived / "basin_signed_tails.csv.gz", dtype={"GAGE_ID": "string"})
    theory = pd.read_csv(derived / "basin_theory_bridge.csv.gz", dtype={"GAGE_ID": "string"})
    prediction = pd.read_csv(
        derived / "basin_prediction_q10.csv.gz", dtype={"GAGE_ID": "string"}
    )
    spatial_bootstrap = pd.read_csv(derived / "prediction_skill_summary.csv")
    water_year_bootstrap = pd.read_csv(
        derived / "prediction_water_year_summary.csv"
    )
    place = pd.read_csv(
        derived / "basin_place_metrics.csv.gz",
        dtype={"GAGE_ID": "string", "HUC02": "string"},
    )

    estimable = scaling[pd.to_numeric(
        scaling["recession_variance_exponent"], errors="coerce"
    ).notna()].copy()
    m = estimable["recession_variance_exponent"].astype(float)
    claims: dict[str, float] = {
        "source_gages": float(len(sample)),
        "accepted_basins": float(truthy(sample["accepted"]).sum()),
        "estimable_basins": float(truthy(sample["estimable"]).sum()),
        "primary_transitions": float(estimable["recession_n_transitions"].sum()),
        "m_median": float(m.median()),
        "m_q25": float(m.quantile(0.25)),
        "m_q75": float(m.quantile(0.75)),
        "m_median_r2": float(estimable["recession_variance_r2"].median()),
        "tail_fraction_median": float(distribution["standardized_tail_fraction"].median()),
        "tail_multiple_median": float(
            distribution["standardized_tail_fraction"].median()
            / distribution["normal_tail_reference"].iloc[0]
        ),
        "excess_kurtosis_median": float(
            distribution["standardized_excess_kurtosis"].median()
        ),
        "positive_tail_multiple_median": float(
            signed["positive_tail_multiple_of_normal"].median()
        ),
        "negative_tail_multiple_median": float(
            signed["negative_tail_multiple_of_normal"].median()
        ),
    }
    claims.update(random_effects(estimable))

    paired = theory[theory["m_decline_recomputed"].notna()].copy()
    adequate = paired[truthy(paired["water_year_bootstrap_sufficient"])].copy()
    claims.update({
        "theory_pair_basins": float(len(paired)),
        "decline_m_median": float(paired["m_decline_recomputed"].median()),
        "two_b_median": float(paired["two_b_recomputed"].median()),
        "decline_minus_two_b_median": float(
            paired["delta_m_decline_minus_2b"].median()
        ),
        "gamma_median": float(paired["gamma_direct"].median()),
        "gamma_ci_below_zero_fraction": float(
            (adequate["gamma_direct_wy_ci_high"] < 0).mean()
        ),
        "negative_within_share_median_paired": float(
            paired["weighted_negative_within_share"].median()
        ),
    })

    global_gaussian = "regularized_power_gaussian"
    state_gaussian = "regularized_power_state_gaussian"
    empirical = "regularized_power_state_empirical"
    for lead in (1, 2, 3):
        claims[f"state_gaussian_crps_gain_{lead}d"] = relative_crps_skill(
            prediction, state_gaussian, global_gaussian, lead
        )
    for lead in (1, 2, 3, 5, 7):
        claims[f"empirical_shape_crps_gain_{lead}d"] = relative_crps_skill(
            prediction, empirical, state_gaussian, lead
        )

    config = json.loads(
        (REPO_ROOT / "config" / "full_study.json").read_text(encoding="utf-8")
    )
    minimum_basins = int(
        config["predictability_test"]["minimum_basins_per_lead_for_claim"]
    )
    minimum_groups = int(
        config["predictability_test"]["minimum_spatial_groups_per_lead_for_claim"]
    )
    bootstrap_tables = {
        "spatial": spatial_bootstrap,
        "water_year": water_year_bootstrap,
    }
    comparisons = {
        "state": (state_gaussian, global_gaussian, (1, 2, 3, 5, 7, 10)),
        "empirical": (empirical, state_gaussian, (1, 2, 3, 5, 7, 10)),
    }
    for bootstrap_name, summary in bootstrap_tables.items():
        for comparison_name, (model, reference, leads) in comparisons.items():
            for lead in leads:
                claims[
                    f"{comparison_name}_{bootstrap_name}_supported_{lead}d"
                ] = bootstrap_supported(
                    summary,
                    model,
                    reference,
                    lead,
                    minimum_basins,
                    minimum_groups,
                )

    regional = summarize_huc02(place).set_index("region")
    positive_medians = regional["positive_tail_multiple_of_normal_median"]
    negative_medians = regional["negative_tail_multiple_of_normal_median"]
    claims.update({
        "lambda_pacific_northwest_n": regional.at[
            "Pacific Northwest", "lambda_star_n"
        ],
        "lambda_pacific_northwest_median": regional.at[
            "Pacific Northwest", "lambda_star_median"
        ],
        "lambda_texas_gulf_n": regional.at["Texas-Gulf", "lambda_star_n"],
        "lambda_texas_gulf_median": regional.at[
            "Texas-Gulf", "lambda_star_median"
        ],
        "gamma_pacific_northwest_n": regional.at[
            "Pacific Northwest", "gamma_direct_n"
        ],
        "gamma_pacific_northwest_median": regional.at[
            "Pacific Northwest", "gamma_direct_median"
        ],
        "gamma_texas_gulf_n": regional.at["Texas-Gulf", "gamma_direct_n"],
        "gamma_texas_gulf_median": regional.at[
            "Texas-Gulf", "gamma_direct_median"
        ],
        "negative_share_south_atlantic_gulf_n": regional.at[
            "South Atlantic-Gulf", "weighted_negative_within_share_n"
        ],
        "negative_share_south_atlantic_gulf_median": regional.at[
            "South Atlantic-Gulf", "weighted_negative_within_share_median"
        ],
        "negative_share_pacific_northwest_n": regional.at[
            "Pacific Northwest", "weighted_negative_within_share_n"
        ],
        "negative_share_pacific_northwest_median": regional.at[
            "Pacific Northwest", "weighted_negative_within_share_median"
        ],
        "regional_positive_tail_median_min": float(positive_medians.min()),
        "regional_positive_tail_median_max": float(positive_medians.max()),
        "regional_negative_tail_median_min": float(negative_medians.min()),
        "regional_negative_tail_median_max": float(negative_medians.max()),
        "regional_all_tail_medians_above_one": float(
            (positive_medians.gt(1) & negative_medians.gt(1)).all()
        ),
        "prediction_california_n": regional.at[
            "California", "relative_crps_gain_percent_n"
        ],
        "prediction_california_median_percent": regional.at[
            "California", "relative_crps_gain_percent_median"
        ],
        "prediction_upper_colorado_n": regional.at[
            "Upper Colorado", "relative_crps_gain_percent_n"
        ],
        "prediction_upper_colorado_median_percent": regional.at[
            "Upper Colorado", "relative_crps_gain_percent_median"
        ],
        "prediction_south_atlantic_gulf_n": regional.at[
            "South Atlantic-Gulf", "relative_crps_gain_percent_n"
        ],
        "prediction_south_atlantic_gulf_median_percent": regional.at[
            "South Atlantic-Gulf", "relative_crps_gain_percent_median"
        ],
        "prediction_ohio_n": regional.at[
            "Ohio", "relative_crps_gain_percent_n"
        ],
        "prediction_ohio_median_percent": regional.at[
            "Ohio", "relative_crps_gain_percent_median"
        ],
    })
    return claims


def verify(output: Path | None = None) -> pd.DataFrame:
    expected = pd.read_csv(REPO_ROOT / "data" / "expected" / "reported_claims.csv")
    actual = compute_claims()
    expected["actual"] = expected["claim_id"].map(actual)
    expected["absolute_difference"] = (expected["actual"] - expected["expected"]).abs()
    expected["passed"] = expected["absolute_difference"] <= expected["tolerance"]
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        expected.to_csv(output / "reported_claim_verification.csv", index=False)
        place = pd.read_csv(
            REPO_ROOT / "data" / "derived" / "basin_place_metrics.csv.gz",
            dtype={"GAGE_ID": "string", "HUC02": "string"},
        )
        summarize_huc02(place).to_csv(output / "huc02_summary.csv", index=False)
        payload = {
            "claims": len(expected),
            "passed": int(expected["passed"].sum()),
            "all_passed": bool(expected["passed"].all()),
            "failed_claims": expected.loc[~expected["passed"], "claim_id"].tolist(),
        }
        (output / "verification_summary.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "verification")
    args = parser.parse_args()
    table = verify(args.output.resolve())
    shown = table[["claim_id", "expected", "actual", "absolute_difference", "passed"]]
    print(shown.to_string(index=False))
    if not table["passed"].all():
        failed = ", ".join(table.loc[~table["passed"], "claim_id"])
        raise SystemExit(f"Verification failed for: {failed}")
    print(f"\nAll {len(table)} registered claims passed.")


if __name__ == "__main__":
    main()
