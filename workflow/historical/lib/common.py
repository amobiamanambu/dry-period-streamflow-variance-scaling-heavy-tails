"""Common paths, logging, receipts, validation, and reproducibility helpers."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPT_ROOT.parent
DEFAULT_CONFIG = SCRIPT_ROOT / "config.json"


STAGES = {
    1: "01_temperature_download",
    2: "02_temperature_validation",
    3: "03_basin_temperature",
    4: "04_climate_database",
    5: "05_discharge_database",
    6: "06_basin_index",
    7: "07_matched_daily",
    8: "08_matched_qc",
    9: "09_snow_proxy",
    10: "10_transitions",
    11: "11_validation_sample",
    12: "12_estimator_validation",
    13: "13_irreversibility",
    14: "14_recession",
    15: "15_continental_results",
    16: "16_scaling_tests",
    17: "17_distribution_tests",
    18: "18_lowflow_predictability_horizons",
    19: "19_paper_figures",
    20: "20_results_archive",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(PROJECT_ROOT)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    start = cfg["project"]["start_date"]
    end = cfg["project"]["end_date"]
    if start > end:
        raise ValueError("project.start_date must not be after project.end_date")
    years = cfg["gridmet"]["years"]
    if len(years) != 2 or years[0] > years[1]:
        raise ValueError("gridmet.years must be [first_year, last_year]")
    snow = cfg["snow_proxy"]
    if snow["snow_temperature_c"] >= snow["rain_temperature_c"]:
        raise ValueError("snow_temperature_c must be below rain_temperature_c")
    if not str(cfg["predictability_test"]["spatial_group_column"]).strip():
        raise ValueError("predictability_test.spatial_group_column must not be empty")
    scaling = cfg["scaling_test"]
    if scaling["target_exponent"] <= 0 or scaling["near_target_tolerance"] <= 0:
        raise ValueError("scaling_test target and tolerance must be positive")
    if not 0 < scaling["minimum_subgroup_estimable_fraction"] <= 1:
        raise ValueError("scaling_test.minimum_subgroup_estimable_fraction must be in (0, 1]")
    distribution = cfg["distribution_test"]
    if distribution["minimum_transitions"] < 100:
        raise ValueError("distribution_test.minimum_transitions must be at least 100")
    if not 0 < distribution["minimum_paired_nonestimated_fraction"] <= 1:
        raise ValueError(
            "distribution_test.minimum_paired_nonestimated_fraction must be in (0, 1]"
        )
    if int(distribution["tail_sensitivity_minimum_transitions"]) < 100:
        raise ValueError(
            "distribution_test.tail_sensitivity_minimum_transitions must be at least 100"
        )
    if not 0 < float(distribution["tail_sensitivity_low_flow_quantile"]) < 0.5:
        raise ValueError(
            "distribution_test.tail_sensitivity_low_flow_quantile must be in (0, 0.5)"
        )
    if not 0 < float(distribution["minimum_tail_sensitivity_basin_fraction"]) <= 1:
        raise ValueError(
            "distribution_test.minimum_tail_sensitivity_basin_fraction must be in (0, 1]"
        )
    if float(distribution["minimum_tail_frequency_multiple_of_gaussian"]) <= 1:
        raise ValueError(
            "distribution_test.minimum_tail_frequency_multiple_of_gaussian must exceed 1"
        )
    prediction = cfg["predictability_test"]
    leads = [int(item) for item in prediction["lead_days"]]
    if not leads or any(item <= 0 for item in leads) or leads != sorted(set(leads)):
        raise ValueError(
            "predictability_test.lead_days must be unique, increasing positive integers"
        )
    quantiles = [float(item) for item in prediction["low_flow_quantiles"]]
    if not quantiles or len(quantiles) != len(set(quantiles)) or any(
        item <= 0 or item >= 0.5 for item in quantiles
    ):
        raise ValueError(
            "predictability_test.low_flow_quantiles must be unique values in (0, 0.5)"
        )
    if float(prediction["primary_low_flow_quantile"]) not in quantiles:
        raise ValueError(
            "predictability_test.primary_low_flow_quantile must occur in low_flow_quantiles"
        )
    if int(prediction["ensemble_members"]) < 21 or int(prediction["ensemble_members"]) % 2 == 0:
        raise ValueError(
            "predictability_test.ensemble_members must be an odd integer of at least 21"
        )
    prior_bounds = [float(item) for item in prediction["variance_exponent_bounds"]]
    if len(prior_bounds) != 2 or prior_bounds[0] >= prior_bounds[1]:
        raise ValueError("predictability_test.variance_exponent_bounds must be [low, high]")
    if float(prediction["variance_exponent_prior_sd"]) <= 0:
        raise ValueError("predictability_test.variance_exponent_prior_sd must be positive")
    if int(prediction["minimum_basins_per_lead_for_claim"]) < 1:
        raise ValueError("predictability_test.minimum_basins_per_lead_for_claim must be positive")
    if int(prediction["water_year_block_bootstrap_replicates"]) < 100:
        raise ValueError(
            "predictability_test.water_year_block_bootstrap_replicates must be at least 100"
        )
    if int(prediction["minimum_evaluation_water_years_per_basin"]) < 2:
        raise ValueError(
            "predictability_test.minimum_evaluation_water_years_per_basin must be at least 2"
        )
    if int(prediction["minimum_spatial_groups_per_lead_for_claim"]) < 1:
        raise ValueError(
            "predictability_test.minimum_spatial_groups_per_lead_for_claim must be positive"
        )


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(cfg["_project_root"]) / path


def output_root(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, cfg["outputs"]["root"])


def stage_dir(cfg: dict[str, Any], number: int, create: bool = True) -> Path:
    if number not in STAGES:
        raise KeyError(f"Unknown stage: {number}")
    path = output_root(cfg) / STAGES[number]
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def database_path(cfg: dict[str, Any]) -> Path:
    path = resolve_path(cfg, cfg["outputs"]["database"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_gage_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(8) if digits else ""


def parse_args(description: str, stage: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON configuration.")
    parser.add_argument("--force", action="store_true", help="Replace this stage's existing outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional basin/file limit for a smoke run.")
    parser.add_argument("--workers", type=int, default=1, help="Independent basin workers for analysis stages.")
    parser.add_argument("--stage", type=int, default=stage, help=argparse.SUPPRESS)
    return parser.parse_args()


def configure_logging(cfg: dict[str, Any], stage: int) -> logging.Logger:
    log_dir = output_root(cfg) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"continental.stage{stage:02d}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / f"{stage:02d}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def success_path(cfg: dict[str, Any], stage: int) -> Path:
    return stage_dir(cfg, stage) / "_SUCCESS.json"


def require_stage(cfg: dict[str, Any], stage: int) -> dict[str, Any]:
    path = success_path(cfg, stage)
    if not path.exists():
        raise RuntimeError(
            f"Stage {stage:02d} is incomplete. Run scripts/{stage:02d}_*.py first. "
            f"Missing receipt: {path}"
        )
    return read_json(path)


def refuse_overwrite(cfg: dict[str, Any], stage: int, force: bool) -> None:
    receipt = success_path(cfg, stage)
    if receipt.exists() and not force:
        raise RuntimeError(
            f"Stage {stage:02d} already completed at {receipt}. "
            "Use --force only if you intend to replace its products."
        )


def write_receipt(
    cfg: dict[str, Any], stage: int, outputs: Iterable[str | Path], metrics: dict[str, Any] | None = None
) -> Path:
    existing = [Path(item).resolve() for item in outputs if Path(item).exists()]
    code_files = list(SCRIPT_ROOT.glob(f"{stage:02d}_*.py"))
    code_files.extend([
        SCRIPT_ROOT / "lib" / "analysis.py",
        SCRIPT_ROOT / "lib" / "stochastic.py",
        SCRIPT_ROOT / "lib" / "statistics.py",
        SCRIPT_ROOT / "lib" / "hydrology.py",
        SCRIPT_ROOT / "lib" / "common.py",
    ])
    code_hashes = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in code_files if path.exists()
    }
    payload = {
        "stage": stage,
        "stage_name": STAGES[stage],
        "completed_utc": utc_now(),
        "config": str(Path(cfg["_config_path"]).resolve()),
        "config_sha256": sha256_file(Path(cfg["_config_path"])),
        "code_sha256": code_hashes,
        "outputs": [str(item) for item in existing],
        "metrics": metrics or {},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    path = success_path(cfg, stage)
    atomic_json(path, payload)
    return path


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with atomic_target(target) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")


@contextlib.contextmanager
def atomic_target(target: Path) -> Iterator[Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def prepare_stage(cfg: dict[str, Any], stage: int, force: bool) -> Path:
    directory = stage_dir(cfg, stage)
    refuse_overwrite(cfg, stage, force)
    if force:
        receipt = success_path(cfg, stage)
        receipt.unlink(missing_ok=True)
    return directory


def copy_provenance(cfg: dict[str, Any]) -> Path:
    destination = output_root(cfg) / "provenance"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(cfg["_config_path"]), destination / "config.used.json")
    return destination


def year_range(cfg: dict[str, Any]) -> range:
    first, last = cfg["gridmet"]["years"]
    return range(int(first), int(last) + 1)


def chunked(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    bucket: list[Any] = []
    for item in items:
        bucket.append(item)
        if len(bucket) == size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def dependency_versions(names: Iterable[str]) -> dict[str, str]:
    from importlib import metadata

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "NOT INSTALLED"
    return versions
