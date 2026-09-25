#!/usr/bin/env python3
"""Run the versioned sensitivity analyses used by the reported study."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / "workflow"
DEFAULT_CONFIG = REPO_ROOT / "config" / "full_study.json"
sys.path.insert(0, str(WORKFLOW))

from lib.common import load_config, output_root, require_stage, sha256_file  # noqa: E402

ANALYSES = {
    "tail-direction": (
        "sensitivity_tail_direction_clustering.py",
        True,
        True,
    ),
    "robust-scale": ("sensitivity_robust_scale.py", True, True),
    "theory-bridge": ("sensitivity_recession_theory_bridge.py", True, True),
    "selection-audit": ("sensitivity_selection_intermittency.py", False, False),
    "mechanism-error": ("sensitivity_recession_measurement_error.py", False, False),
    "shrinkage": ("sensitivity_exponent_shrinkage.py", True, True),
    "signed-tail-attributes": (
        "sensitivity_signed_tail_attributes.py",
        False,
        True,
    ),
}

STAGE_DEPENDENCIES = {
    "tail-direction": (10, 15, 17),
    "robust-scale": (10, 17),
    "theory-bridge": (10, 14, 15),
    "selection-audit": (15, 17),
    "mechanism-error": (14, 15, 17),
    "shrinkage": (10, 15, 18),
    "signed-tail-attributes": (15,),
}

ANALYSIS_DEPENDENCIES = {
    "robust-scale": ("tail-direction",),
    "selection-audit": ("tail-direction",),
    "signed-tail-attributes": ("tail-direction",),
}


def dependency_closure(names: set[str]) -> set[str]:
    """Return requested sensitivity analyses plus their prerequisites."""
    expanded = set(names)
    pending = list(names)
    while pending:
        name = pending.pop()
        for dependency in ANALYSIS_DEPENDENCIES.get(name, ()):
            if dependency not in expanded:
                expanded.add(dependency)
                pending.append(dependency)
    return expanded


def _resolve_receipt_path(cfg: dict, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(cfg["_project_root"]) / path


def validate_tail_direction_receipt(cfg: dict) -> Path:
    """Validate the shared signed-tail prerequisite before dependent analyses."""
    directory = output_root(cfg) / "sensitivity_analyses" / "tail_direction"
    receipt_path = directory / "_SUCCESS.json"
    if not receipt_path.is_file():
        raise RuntimeError(
            "The tail-direction prerequisite is missing. "
            "Run it before dependent sensitivity analyses."
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    config_path = Path(cfg["_config_path"])
    if receipt.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError(
            "The tail-direction prerequisite used a different configuration; "
            "rerun it with --force."
        )

    script_path = WORKFLOW / ANALYSES["tail-direction"][0]
    recorded_code = set(receipt.get("code_sha256", {}).values())
    if sha256_file(script_path) not in recorded_code:
        raise RuntimeError(
            "The tail-direction prerequisite used different code; rerun it with --force."
        )

    for value, recorded_hash in receipt.get("frozen_input_receipts_sha256", {}).items():
        path = _resolve_receipt_path(cfg, value)
        if not path.is_file() or sha256_file(path) != recorded_hash:
            raise RuntimeError(
                "A numbered-stage receipt used by the tail-direction analysis changed: "
                f"{path}. Rerun the tail-direction analysis with --force."
            )

    stage15_attributes = output_root(cfg) / "15_continental_results" / "continental_basin_results.csv"
    expected_stage15_hash = receipt.get("frozen_stage15_attributes_sha256")
    if (
        not stage15_attributes.is_file()
        or sha256_file(stage15_attributes) != expected_stage15_hash
    ):
        raise RuntimeError(
            "The Stage-15 attributes used by the tail-direction analysis changed; "
            "rerun it with --force."
        )

    stage17_diagnostics = output_root(cfg) / "17_distribution_tests" / "distribution_basin_diagnostics.csv"
    expected_stage17_hash = receipt.get("frozen_stage17_diagnostics_sha256")
    if (
        not stage17_diagnostics.is_file()
        or sha256_file(stage17_diagnostics) != expected_stage17_hash
    ):
        raise RuntimeError(
            "The Stage-17 diagnostics used by the tail-direction analysis changed; "
            "rerun it with --force."
        )

    hashes = receipt.get("output_sha256", {})
    if not hashes:
        raise RuntimeError("The tail-direction receipt contains no output hashes.")
    for filename, recorded_hash in hashes.items():
        path = directory / filename
        if not path.is_file() or sha256_file(path) != recorded_hash:
            raise RuntimeError(
                f"Tail-direction output differs from its receipt: {path}. "
                "Rerun it with --force."
            )
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ANALYSES,
        default=None,
        help="Run only the named analyses, in dependency-safe canonical order.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    requested = set(args.only or ANALYSES)
    selected = dependency_closure(requested)
    cfg = load_config(args.config)

    required_stages = sorted(
        {stage for name in selected for stage in STAGE_DEPENDENCIES[name]}
    )
    for stage in required_stages:
        require_stage(cfg, stage)

    for name, (filename, accepts_workers, accepts_force) in ANALYSES.items():
        if name not in selected:
            continue
        is_implicit_prerequisite = name not in requested
        if name == "tail-direction" and is_implicit_prerequisite and not args.force:
            receipt = output_root(cfg) / "sensitivity_analyses" / "tail_direction" / "_SUCCESS.json"
            if receipt.exists():
                validate_tail_direction_receipt(cfg)
                print(
                    "\n[sensitivity analysis: tail-direction] "
                    "validated existing dependency-safe result",
                    flush=True,
                )
                continue
        for dependency in ANALYSIS_DEPENDENCIES.get(name, ()):
            if dependency == "tail-direction":
                validate_tail_direction_receipt(cfg)
        command = [
            sys.executable,
            str(WORKFLOW / filename),
            "--config",
            str(args.config.expanduser().resolve()),
        ]
        if accepts_workers:
            command.extend(["--workers", str(args.workers)])
        if args.force and accepts_force:
            command.append("--force")
        print(f"\n[sensitivity analysis: {name}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        if name == "tail-direction":
            validate_tail_direction_receipt(cfg)

        # Recheck shared prerequisites after a dependent run so a concurrent
        # change cannot leave a result that appears dependency-safe.
        for dependency in ANALYSIS_DEPENDENCIES.get(name, ()):
            if dependency == "tail-direction":
                validate_tail_direction_receipt(cfg)


if __name__ == "__main__":
    main()
