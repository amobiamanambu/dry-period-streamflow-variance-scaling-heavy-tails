from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / "workflow"
sys.path.insert(0, str(WORKFLOW))

from lib.common import (  # noqa: E402
    fingerprint_output,
    prepare_stage,
    require_stage,
    sha256_file,
    write_receipt,
)
from lib.hydrology import add_temperature_index_snow_proxy, add_transition_columns  # noqa: E402


def load_stage05():
    spec = importlib.util.spec_from_file_location(
        "workflow_stage05", WORKFLOW / "05_ingest_usgs_discharge.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Stage 05 for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE05 = load_stage05()


def load_sensitivity_runner():
    spec = importlib.util.spec_from_file_location(
        "sensitivity_runner", REPO_ROOT / "reproducibility" / "run_sensitivity_analyses.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load sensitivity runner for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SENSITIVITY_RUNNER = load_sensitivity_runner()


class DischargeManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.batch = self.root / "batch_00001.rdb.gz"
        self.batch.write_bytes(b"frozen-usgs-batch")
        self.digest = hashlib.sha256(self.batch.read_bytes()).hexdigest()
        self.manifest = self.root / "manifest.csv"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_manifest(self, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_csv(self.manifest, index=False)

    def valid_row(self) -> dict[str, object]:
        return {
            "raw_file": self.batch.name,
            "bytes_gz": self.batch.stat().st_size,
            "sha256_gz": self.digest,
        }

    def test_manifest_accepts_matching_batch(self) -> None:
        self.write_manifest([self.valid_row()])
        result = STAGE05.verify_download_manifest([self.batch], self.manifest)
        self.assertEqual(result["raw_file"].tolist(), [self.batch.name])

    def test_manifest_rejects_unlisted_batch(self) -> None:
        row = self.valid_row()
        row["raw_file"] = "different.rdb.gz"
        self.write_manifest([row])
        with self.assertRaisesRegex(RuntimeError, "not listed in manifest"):
            STAGE05.verify_download_manifest([self.batch], self.manifest)

    def test_manifest_rejects_byte_count_mismatch(self) -> None:
        row = self.valid_row()
        row["bytes_gz"] = self.batch.stat().st_size + 1
        self.write_manifest([row])
        with self.assertRaisesRegex(RuntimeError, "byte count mismatch"):
            STAGE05.verify_download_manifest([self.batch], self.manifest)

    def test_manifest_rejects_hash_mismatch(self) -> None:
        row = self.valid_row()
        row["sha256_gz"] = "0" * 64
        self.write_manifest([row])
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            STAGE05.verify_download_manifest([self.batch], self.manifest)

    def test_manifest_rejects_duplicate_file_rows(self) -> None:
        row = self.valid_row()
        self.write_manifest([row, row])
        with self.assertRaisesRegex(ValueError, "duplicate raw_file"):
            STAGE05.verify_download_manifest([self.batch], self.manifest)


class SharedDatabaseRerunTests(unittest.TestCase):
    def test_stage05_force_preflight_rejects_corrupted_shared_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            output_root = root / "outputs"
            stage = output_root / "05_discharge_database"
            stage.mkdir(parents=True)
            database = output_root / "data" / "continental_daily.sqlite"
            database.parent.mkdir(parents=True)
            database.write_bytes(b"verified-climate-and-discharge")
            audit = stage / "discharge_ingest_audit.csv"
            audit.write_text("status\ncomplete\n", encoding="utf-8")
            receipt_path = stage / "_SUCCESS.json"
            outputs = [str(database.resolve()), str(audit.resolve())]
            receipt = {
                "receipt_schema": 2,
                "stage": 5,
                "outputs": outputs,
                "output_fingerprints": {
                    value: fingerprint_output(value, excluded_paths={receipt_path.resolve()})
                    for value in outputs
                },
            }
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            cfg = {
                "_config_path": str(config_path),
                "_project_root": str(REPO_ROOT),
                "outputs": {
                    "root": str(output_root),
                    "database": str(database),
                },
            }

            self.assertTrue(STAGE05.validate_existing_stage5_before_rerun(cfg))
            database.write_bytes(b"externally-corrupted-climate-table")

            with self.assertRaisesRegex(RuntimeError, "outputs changed"):
                STAGE05.validate_existing_stage5_before_rerun(cfg)


class StagePreparationTests(unittest.TestCase):
    def test_force_removes_stale_stage_products(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg = {
                "_project_root": str(REPO_ROOT),
                "outputs": {"root": temporary},
            }
            stage = Path(temporary) / "05_discharge_database"
            nested = stage / "stale" / "basin.csv"
            nested.parent.mkdir(parents=True)
            nested.write_text("stale\n", encoding="utf-8")
            (stage / "_SUCCESS.json").write_text("{}\n", encoding="utf-8")

            output = prepare_stage(cfg, 5, force=True)

            self.assertEqual(output, stage)
            self.assertEqual(list(stage.iterdir()), [])


class StageReceiptIntegrityTests(unittest.TestCase):
    def make_config(self, temporary: str) -> dict[str, object]:
        config_path = Path(temporary) / "config.json"
        config_path.write_text("{}\n", encoding="utf-8")
        return {
            "_config_path": str(config_path),
            "_project_root": str(REPO_ROOT),
            "outputs": {"root": str(Path(temporary) / "outputs")},
        }

    def test_require_stage_rejects_mutated_nested_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg = self.make_config(temporary)
            product = Path(temporary) / "outputs" / "01_temperature_download" / "data"
            product.mkdir(parents=True)
            nested = product / "values.csv"
            nested.write_text("value\n1\n", encoding="utf-8")
            write_receipt(cfg, 1, [product])

            require_stage(cfg, 1)
            nested.write_text("value\n999\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "outputs changed"):
                require_stage(cfg, 1)

    def test_write_receipt_rejects_a_missing_declared_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg = self.make_config(temporary)
            missing = Path(temporary) / "outputs" / "01_temperature_download" / "missing.csv"

            with self.assertRaisesRegex(RuntimeError, "declared outputs are missing"):
                write_receipt(cfg, 1, [missing])

            receipt = Path(temporary) / "outputs" / "01_temperature_download" / "_SUCCESS.json"
            self.assertFalse(receipt.exists())

    def test_legacy_receipt_keeps_existence_only_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg = self.make_config(temporary)
            stage = Path(temporary) / "outputs" / "01_temperature_download"
            stage.mkdir(parents=True)
            product = stage / "legacy.txt"
            product.write_text("legacy\n", encoding="utf-8")
            config_path = Path(str(cfg["_config_path"]))
            receipt = {
                "stage": 1,
                "outputs": [str(product)],
                "config_sha256": sha256_file(config_path),
                "code_sha256": {},
            }
            (stage / "_SUCCESS.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )

            self.assertEqual(require_stage(cfg, 1)["stage"], 1)

    def test_require_stage_does_not_rehash_ancestor_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg = self.make_config(temporary)
            stage_one = Path(temporary) / "outputs" / "01_temperature_download"
            stage_one.mkdir(parents=True)
            first_product = stage_one / "first.txt"
            first_product.write_text("first\n", encoding="utf-8")
            write_receipt(cfg, 1, [first_product])

            stage_two = Path(temporary) / "outputs" / "02_temperature_validation"
            stage_two.mkdir(parents=True)
            second_product = stage_two / "second.txt"
            second_product.write_text("second\n", encoding="utf-8")
            write_receipt(cfg, 2, [second_product])

            with mock.patch(
                "lib.common.fingerprint_output",
                wraps=sys.modules["lib.common"].fingerprint_output,
            ) as fingerprint:
                require_stage(cfg, 2)

            checked = [Path(call.args[0]).resolve() for call in fingerprint.call_args_list]
            self.assertEqual(checked, [second_product.resolve()])

    def test_downstream_receipt_allows_documented_shared_output_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg = self.make_config(temporary)
            shared = Path(temporary) / "shared.sqlite"
            shared.write_bytes(b"stage-one")
            write_receipt(cfg, 1, [shared])

            stage_two = Path(temporary) / "outputs" / "02_temperature_validation"
            stage_two.mkdir(parents=True)
            second_product = stage_two / "audit.csv"
            shared.write_bytes(b"stage-two-legitimate-mutation")
            second_product.write_text("status\ncomplete\n", encoding="utf-8")
            write_receipt(cfg, 2, [shared, second_product])

            self.assertEqual(require_stage(cfg, 2)["stage"], 2)
            with self.assertRaisesRegex(RuntimeError, "outputs changed"):
                require_stage(cfg, 1)


class MissingForcingScreenTests(unittest.TestCase):
    def test_missing_temperature_excludes_every_affected_dry_window(self) -> None:
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        daily = pd.DataFrame(
            {
                "date": dates,
                "q_mm_day": [1.0, 0.9, 0.8, 0.7, 0.6],
                "pr_mm": [0.0] * 5,
                "tmean_c": [2.0, 2.0, float("nan"), 2.0, 2.0],
            }
        )
        snow_settings = {
            "snow_temperature_c": -1.0,
            "rain_temperature_c": 3.0,
            "melt_temperature_c": 0.0,
            "degree_day_factor_mm_c_day": 3.0,
            "minimum_snow_storage_mm": 0.1,
        }
        transition_settings = {
            "minimum_q_mm_day": 1e-6,
            "recession_lags_days": [1],
            "antecedent_windows_days": [1],
            "precipitation_thresholds_mm_day": [0.0],
        }

        screened = add_temperature_index_snow_proxy(daily, snow_settings)
        transitions = add_transition_columns(screened, transition_settings)

        self.assertTrue(pd.isna(screened.loc[2, "snowmelt_risk"]))
        self.assertTrue(bool(transitions.loc[1, "forcing_window_melt_a1_l1"]))
        self.assertTrue(bool(transitions.loc[2, "forcing_window_melt_a1_l1"]))
        self.assertFalse(bool(transitions.loc[1, "dry_p0p0_a1_l1"]))
        self.assertFalse(bool(transitions.loc[2, "dry_p0p0_a1_l1"]))
        self.assertTrue(bool(transitions.loc[0, "dry_p0p0_a1_l1"]))


class SensitivityDependencyTests(unittest.TestCase):
    def test_robust_scale_adds_tail_direction_prerequisite(self) -> None:
        selected = SENSITIVITY_RUNNER.dependency_closure({"robust-scale"})
        self.assertEqual(selected, {"tail-direction", "robust-scale"})

    def test_independent_analysis_does_not_add_unrelated_sidecars(self) -> None:
        selected = SENSITIVITY_RUNNER.dependency_closure({"theory-bridge"})
        self.assertEqual(selected, {"theory-bridge"})

    def test_tail_dependency_rejects_changed_stage15_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            output = root / "run"
            cfg = {
                "_config_path": str(config_path),
                "_project_root": str(REPO_ROOT),
                "outputs": {"root": str(output)},
            }

            receipt_paths = {}
            for number, directory in (
                (10, "10_transitions"),
                (15, "15_continental_results"),
                (17, "17_distribution_tests"),
            ):
                stage = output / directory
                stage.mkdir(parents=True)
                receipt = stage / "_SUCCESS.json"
                receipt.write_text(f'{{"stage": {number}}}\n', encoding="utf-8")
                receipt_paths[number] = receipt

            stage15_table = output / "15_continental_results" / "continental_basin_results.csv"
            stage15_table.write_text("GAGE_ID\n01234567\n", encoding="utf-8")
            stage17_table = output / "17_distribution_tests" / "distribution_basin_diagnostics.csv"
            stage17_table.write_text("GAGE_ID\n01234567\n", encoding="utf-8")

            tail = output / "sensitivity_analyses" / "tail_direction"
            tail.mkdir(parents=True)
            product = tail / "basin_tail_direction_and_clustering.csv"
            product.write_text("GAGE_ID\n01234567\n", encoding="utf-8")
            script = WORKFLOW / "sensitivity_tail_direction_clustering.py"
            payload = {
                "config_sha256": sha256_file(config_path),
                "code_sha256": {"workflow/sensitivity_tail_direction_clustering.py": sha256_file(script)},
                "frozen_input_receipts_sha256": {
                    str(path): sha256_file(path) for path in receipt_paths.values()
                },
                "frozen_stage15_attributes_sha256": sha256_file(stage15_table),
                "frozen_stage17_diagnostics_sha256": sha256_file(stage17_table),
                "output_sha256": {product.name: sha256_file(product)},
            }
            (tail / "_SUCCESS.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            SENSITIVITY_RUNNER.validate_tail_direction_receipt(cfg)
            receipt_paths[15].write_text('{"stage": 15, "changed": true}\n', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "numbered-stage receipt"):
                SENSITIVITY_RUNNER.validate_tail_direction_receipt(cfg)


if __name__ == "__main__":
    unittest.main()
