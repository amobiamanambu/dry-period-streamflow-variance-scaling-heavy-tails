#!/usr/bin/env python3
"""Rerun the primary estimator on nine complete real-gage examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import fit_primary_innovations, tail_survival
from example_selection import select_from_release


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "example")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads((REPO_ROOT / "config" / "full_study.json").read_text(encoding="utf-8"))
    transitions = pd.read_csv(
        REPO_ROOT / "data" / "example" / "example_transitions.csv.gz",
        dtype={"GAGE_ID": "string"}, parse_dates=["date"], low_memory=False,
    )
    metadata = pd.read_csv(
        REPO_ROOT / "data" / "example" / "gage_metadata.csv", dtype={"GAGE_ID": "string"}
    )
    transitions["GAGE_ID"] = transitions["GAGE_ID"].str.zfill(8)
    metadata["GAGE_ID"] = metadata["GAGE_ID"].str.zfill(8)

    selection = select_from_release()
    selected_gages = set(selection["GAGE_ID"])
    example_gages = set(transitions["GAGE_ID"].unique())
    if selected_gages != example_gages:
        raise RuntimeError(
            "Released example records do not match the reconstructed regional medoids"
        )
    selection.to_csv(output / "example_gage_selection.csv", index=False)

    result_rows: list[dict] = []
    bin_frames: list[pd.DataFrame] = []
    survival_frames: list[pd.DataFrame] = []
    for gage, frame in transitions.groupby("GAGE_ID", sort=True):
        metrics, bins, retained = fit_primary_innovations(frame.sort_values("date"), config)
        result_rows.append({"GAGE_ID": gage, **metrics})
        bins.insert(0, "GAGE_ID", gage)
        bin_frames.append(bins)
        curve = tail_survival(retained["standardized_innovation"].to_numpy(float))
        curve.insert(0, "GAGE_ID", gage)
        survival_frames.append(curve)

    results = pd.DataFrame(result_rows).merge(
        metadata, on="GAGE_ID", how="left", validate="one_to_one", suffixes=("", "_frozen")
    )
    difference = (
        results["variance_exponent"] - results["recession_variance_exponent"]
    ).abs()
    if difference.max() > 1e-10:
        raise RuntimeError(
            "Example exponent does not reproduce the frozen estimator; "
            f"maximum absolute difference was {difference.max():.3g}"
        )

    results.to_csv(output / "example_basin_results.csv", index=False)
    pd.concat(bin_frames, ignore_index=True).to_csv(
        output / "example_conditional_bins.csv", index=False
    )
    pd.concat(survival_frames, ignore_index=True).to_csv(
        output / "example_tail_survival.csv", index=False
    )
    display = results[[
        "GAGE_ID", "AGGECOREGION", "n_transitions", "variance_exponent",
        "variance_r2", "two_sided_tail_multiple",
    ]].copy()
    print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote demonstration products to {output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
