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


RECEIPT_SCHEMA_VERSION = 2
OUTPUT_FINGERPRINT_ALGORITHM = "sha256-tree-v1"


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


def validate_stage_outputs(
    cfg: dict[str, Any],
    stage: int,
    receipt: dict[str, Any] | None = None,
    *,
    validate_contents: bool = True,
) -> dict[str, Any]:
    """Validate declared products without imposing code or config identity.

    This narrower check is useful immediately before a forced rerun that will
    replace the stage receipt. It prevents an already-corrupted product from
    being used as the starting point merely because the author intentionally
    changed the stage code.
    """
    path = success_path(cfg, stage)
    if receipt is None:
        if not path.exists():
            raise RuntimeError(f"Stage {stage:02d} receipt is missing: {path}")
        receipt = read_json(path)
    if int(receipt.get("stage", -1)) != stage:
        raise RuntimeError(f"Stage {stage:02d} receipt has the wrong stage identifier: {path}")

    missing_outputs = []
    for value in receipt.get("outputs", []):
        output = _resolve_receipt_path(cfg, value)
        if not output.exists():
            missing_outputs.append(str(output))
    if missing_outputs:
        raise RuntimeError(
            f"Stage {stage:02d} receipt exists, but recorded outputs are missing:\n- "
            + "\n- ".join(missing_outputs)
        )

    schema_version = int(receipt.get("receipt_schema", 1))
    output_fingerprints = receipt.get("output_fingerprints")
    if schema_version >= RECEIPT_SCHEMA_VERSION and not isinstance(
        output_fingerprints, dict
    ):
        raise RuntimeError(
            f"Stage {stage:02d} receipt is missing its output fingerprints: {path}"
        )
    if not validate_contents or not isinstance(output_fingerprints, dict):
        return receipt

    recorded_outputs = [str(value) for value in receipt.get("outputs", [])]
    missing_fingerprints = sorted(set(recorded_outputs) - set(output_fingerprints))
    extra_fingerprints = sorted(set(output_fingerprints) - set(recorded_outputs))
    if missing_fingerprints or extra_fingerprints:
        details = []
        if missing_fingerprints:
            details.append("missing: " + ", ".join(missing_fingerprints))
        if extra_fingerprints:
            details.append("unregistered: " + ", ".join(extra_fingerprints))
        raise RuntimeError(
            f"Stage {stage:02d} receipt has an inconsistent output-fingerprint "
            f"registry ({'; '.join(details)}). Rerun it with --force."
        )

    changed_outputs = []
    file_hash_cache: dict[Path, str] = {}
    excluded = {path.resolve()}
    for value in recorded_outputs:
        output = _resolve_receipt_path(cfg, value)
        observed = fingerprint_output(
            output,
            excluded_paths=excluded,
            file_hash_cache=file_hash_cache,
        )
        if observed != output_fingerprints[value]:
            changed_outputs.append(str(output))
    if changed_outputs:
        raise RuntimeError(
            f"Stage {stage:02d} outputs changed after its receipt was written. "
            f"Rerun it from a verified upstream state before continuing:\n- "
            + "\n- ".join(changed_outputs)
        )
    return receipt


def require_stage(
    cfg: dict[str, Any],
    stage: int,
    *,
    validate_outputs: bool = True,
    _visited: set[int] | None = None,
) -> dict[str, Any]:
    """Validate a stage receipt and its dependency chain.

    The requested stage's declared outputs are checked against their recorded
    content fingerprints. Ancestor receipts are still checked for continuity,
    configuration, code, and output existence, but their often-large contents
    are not rehashed on every downstream check. A caller that consumes an
    older stage directly should call ``require_stage`` for that stage as well.

    Receipts written before schema version 2 have no output fingerprints. They
    remain readable and receive the historical existence-only output check.
    """
    if _visited is None:
        _visited = set()
    if stage in _visited:
        raise RuntimeError(f"Cycle detected in receipt chain at Stage {stage:02d}")
    _visited.add(stage)

    path = success_path(cfg, stage)
    if not path.exists():
        raise RuntimeError(
            f"Stage {stage:02d} is incomplete. Run workflow/{stage:02d}_*.py first. "
            f"Missing receipt: {path}"
        )
    receipt = read_json(path)
    if int(receipt.get("stage", -1)) != stage:
        raise RuntimeError(f"Stage {stage:02d} receipt has the wrong stage identifier: {path}")

    current_config_hash = sha256_file(Path(cfg["_config_path"]))
    if receipt.get("config_sha256") != current_config_hash:
        raise RuntimeError(
            f"Stage {stage:02d} used a different configuration. "
            f"Rerun it with --force before continuing."
        )

    validate_stage_outputs(
        cfg,
        stage,
        receipt,
        validate_contents=validate_outputs,
    )

    changed_code = []
    for value, recorded_hash in receipt.get("code_sha256", {}).items():
        code_path = Path(value)
        if not code_path.is_absolute():
            code_path = Path(cfg["_project_root"]) / code_path
        if not code_path.exists() or sha256_file(code_path) != recorded_hash:
            changed_code.append(str(code_path))
    if changed_code:
        raise RuntimeError(
            f"Stage {stage:02d} code differs from its receipt. "
            f"Rerun it with --force before continuing:\n- "
            + "\n- ".join(changed_code)
        )

    if stage > 1:
        upstream = receipt.get("upstream_receipt")
        if not isinstance(upstream, dict):
            raise RuntimeError(
                f"Stage {stage:02d} receipt predates dependency chaining. "
                "Rerun it with --force before continuing."
            )
        upstream_stage = int(upstream.get("stage", -1))
        if upstream_stage != stage - 1:
            raise RuntimeError(
                f"Stage {stage:02d} receipt records an invalid predecessor: "
                f"{upstream_stage:02d}"
            )
        upstream_path = Path(str(upstream.get("path", "")))
        if not upstream_path.is_absolute():
            upstream_path = Path(cfg["_project_root"]) / upstream_path
        if not upstream_path.exists():
            raise RuntimeError(
                f"Stage {stage:02d} predecessor receipt is missing: {upstream_path}"
            )
        if sha256_file(upstream_path) != upstream.get("sha256"):
            raise RuntimeError(
                f"Stage {stage:02d} was built from an older Stage "
                f"{upstream_stage:02d} receipt. Rerun Stage {stage:02d} with --force."
            )
        require_stage(
            cfg,
            upstream_stage,
            validate_outputs=False,
            _visited=_visited,
        )
    return receipt


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
    declared = [Path(item).resolve() for item in outputs]
    missing = [str(item) for item in declared if not item.exists()]
    if missing:
        raise RuntimeError(
            f"Cannot write Stage {stage:02d} receipt because declared outputs are missing:\n- "
            + "\n- ".join(missing)
        )
    existing = list(dict.fromkeys(declared))
    code_files = list(SCRIPT_ROOT.glob(f"{stage:02d}_*.py"))
    code_files.extend(sorted((SCRIPT_ROOT / "lib").glob("*.py")))
    code_hashes = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in code_files if path.exists()
    }
    config_path = Path(cfg["_config_path"]).resolve()
    receipt_path = success_path(cfg, stage).resolve()

    def portable_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path.resolve())

    upstream_record = None
    if stage > 1:
        upstream = success_path(cfg, stage - 1)
        if not upstream.exists():
            raise RuntimeError(
                f"Cannot write Stage {stage:02d} receipt without Stage "
                f"{stage - 1:02d} receipt: {upstream}"
            )
        upstream_record = {
            "stage": stage - 1,
            "path": portable_path(upstream),
            "sha256": sha256_file(upstream),
        }

    portable_outputs = [portable_path(item) for item in existing]
    file_hash_cache: dict[Path, str] = {}
    output_fingerprints = {
        portable: fingerprint_output(
            item,
            excluded_paths={receipt_path},
            file_hash_cache=file_hash_cache,
        )
        for portable, item in zip(portable_outputs, existing, strict=True)
    }

    payload = {
        "receipt_schema": RECEIPT_SCHEMA_VERSION,
        "stage": stage,
        "stage_name": STAGES[stage],
        "completed_utc": utc_now(),
        "config": portable_path(config_path),
        "config_sha256": sha256_file(config_path),
        "code_sha256": code_hashes,
        "outputs": portable_outputs,
        "output_fingerprint_algorithm": OUTPUT_FINGERPRINT_ALGORITHM,
        "output_fingerprints": output_fingerprints,
        "metrics": metrics or {},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if upstream_record is not None:
        payload["upstream_receipt"] = upstream_record
    path = success_path(cfg, stage)
    atomic_json(path, payload)
    return path


def _resolve_receipt_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(cfg["_project_root"]) / path
    return path.resolve()


def _cached_sha256(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.resolve()
    if resolved not in cache:
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Output changed while it was being fingerprinted: {path}")
        cache[resolved] = digest
    return cache[resolved]


def fingerprint_output(
    path: str | Path,
    *,
    excluded_paths: set[Path] | None = None,
    file_hash_cache: dict[Path, str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic content fingerprint for one file or directory."""
    target = Path(path).resolve()
    excluded = {item.resolve() for item in (excluded_paths or set())}
    cache = file_hash_cache if file_hash_cache is not None else {}
    if target in excluded:
        raise ValueError(f"A receipt cannot declare itself as an output: {target}")
    if not target.exists():
        raise FileNotFoundError(target)

    if target.is_file():
        stat = target.stat()
        return {
            "algorithm": "sha256-v1",
            "kind": "file",
            "bytes": int(stat.st_size),
            "sha256": _cached_sha256(target, cache),
        }
    if not target.is_dir():
        raise RuntimeError(f"Unsupported declared output type: {target}")

    tree_digest = hashlib.sha256()
    entry_count = 0
    file_count = 0
    total_bytes = 0
    for child in sorted(target.rglob("*"), key=lambda item: item.relative_to(target).as_posix()):
        resolved_child = child.resolve()
        if resolved_child in excluded:
            continue
        relative = child.relative_to(target).as_posix()
        if child.is_symlink():
            record = {
                "kind": "symlink",
                "path": relative,
                "target": os.readlink(child),
            }
        elif child.is_dir():
            record = {"kind": "directory", "path": relative}
        elif child.is_file():
            stat = child.stat()
            record = {
                "kind": "file",
                "path": relative,
                "bytes": int(stat.st_size),
                "sha256": _cached_sha256(child, cache),
            }
            file_count += 1
            total_bytes += int(stat.st_size)
        else:
            raise RuntimeError(f"Unsupported output entry type: {child}")
        canonical = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        tree_digest.update(canonical)
        tree_digest.update(b"\n")
        entry_count += 1

    return {
        "algorithm": OUTPUT_FINGERPRINT_ALGORITHM,
        "kind": "directory",
        "entries": entry_count,
        "files": file_count,
        "bytes": total_bytes,
        "sha256": tree_digest.hexdigest(),
    }


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
        resolved_directory = directory.resolve()
        resolved_root = output_root(cfg).resolve()
        if resolved_directory.parent != resolved_root:
            raise RuntimeError(
                f"Refusing to clean unexpected stage directory: {resolved_directory}"
            )
        for child in resolved_directory.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
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
