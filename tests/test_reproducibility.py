from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "reproducibility"))

from core import fit_primary_innovations  # noqa: E402
from example_selection import EXPECTED_GAGES, select_from_release  # noqa: E402
from verify_reported_results import verify  # noqa: E402


class ReproducibilityTests(unittest.TestCase):
    def test_registered_claims_pass(self) -> None:
        result = verify()
        self.assertTrue(result["passed"].all())


    def test_example_gage_matches_frozen_exponent(self) -> None:
        transitions = pd.read_csv(
            REPO_ROOT / "data" / "example" / "example_transitions.csv.gz",
            dtype={"GAGE_ID": "string"}, parse_dates=["date"], low_memory=False,
        )
        metadata = pd.read_csv(
            REPO_ROOT / "data" / "example" / "gage_metadata.csv", dtype={"GAGE_ID": "string"}
        )
        transitions["GAGE_ID"] = transitions["GAGE_ID"].str.zfill(8)
        metadata["GAGE_ID"] = metadata["GAGE_ID"].str.zfill(8)
        gage = sorted(transitions["GAGE_ID"].unique())[0]
        config = json.loads(
            (REPO_ROOT / "config" / "full_study.json").read_text(encoding="utf-8")
        )
        metrics, bins, retained = fit_primary_innovations(
            transitions[transitions["GAGE_ID"].eq(gage)].sort_values("date"), config
        )
        expected = float(
            metadata.loc[
                metadata["GAGE_ID"].eq(gage), "recession_variance_exponent"
            ].iloc[0]
        )
        self.assertLess(abs(metrics["variance_exponent"] - expected), 1e-10)
        self.assertGreaterEqual(
            len(bins), config["recession"]["minimum_valid_bins_for_fit"]
        )
        self.assertEqual(len(retained), metrics["n_standardized"])


    def test_station_identifiers_keep_leading_zeroes(self) -> None:
        metadata = pd.read_csv(
            REPO_ROOT / "data" / "example" / "gage_metadata.csv",
            dtype={"GAGE_ID": "string"},
        )
        self.assertTrue(metadata["GAGE_ID"].str.fullmatch(r"\d{8}").all())

    def test_example_gages_reproduce_regional_medoid_rule(self) -> None:
        selected = select_from_release()
        actual = dict(zip(selected["AGGECOREGION"], selected["GAGE_ID"]))
        self.assertEqual(actual, EXPECTED_GAGES)


if __name__ == "__main__":
    unittest.main()
