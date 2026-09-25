#!/usr/bin/env python3
"""Regenerate the non-manuscript README previews and record their inputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "outputs" / "figures"
DESTINATION = REPO_ROOT / "docs" / "figures"
PREVIEWS = (
    "01_real_gage_estimator_check.png",
    "02_continental_result_audit.png",
)
SOURCES = (
    "maintainer/refresh_previews.py",
    "config/full_study.json",
    ".github/workflows/ci.yml",
    "requirements-core.txt",
    "environment.yml",
    "reproducibility/core.py",
    "reproducibility/example_selection.py",
    "reproducibility/run_example.py",
    "reproducibility/verify_reported_results.py",
    "reproducibility/make_provenance_figures.py",
    "workflow/lib/stochastic.py",
    "data/example/example_transitions.csv.gz",
    "data/example/gage_metadata.csv",
    "data/derived/basin_distribution.csv.gz",
    "data/derived/basin_place_metrics.csv.gz",
    "data/derived/basin_prediction_q10.csv.gz",
    "data/derived/basin_sample.csv.gz",
    "data/derived/basin_scaling.csv.gz",
    "data/derived/basin_signed_tails.csv.gz",
    "data/derived/basin_theory_bridge.csv.gz",
    "data/derived/prediction_skill_summary.csv",
    "data/derived/prediction_water_year_summary.csv",
    "data/expected/reported_claims.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for script in (
        "reproducibility/run_example.py",
        "reproducibility/verify_reported_results.py",
        "reproducibility/make_provenance_figures.py",
    ):
        subprocess.run([sys.executable, script], cwd=REPO_ROOT, check=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name in PREVIEWS:
        shutil.copy2(OUTPUT / name, DESTINATION / name)
    payload = {
        "purpose": "Diagnostic provenance previews; not manuscript artwork.",
        "generator": "maintainer/refresh_previews.py",
        "environment_scope": (
            "The versions below describe the maintainer environment that rendered "
            "the committed PNG previews. The tested numerical release runtime is "
            "Python 3.11 as pinned in environment.yml and continuous integration."
        ),
        "generation_environment": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in ("numpy", "pandas", "scipy", "matplotlib")
            },
        },
        "source_sha256": {
            relative: sha256(REPO_ROOT / relative) for relative in SOURCES
        },
        "preview_sha256": {
            f"docs/figures/{name}": sha256(DESTINATION / name)
            for name in PREVIEWS
        },
    }
    (DESTINATION / "PROVENANCE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Refreshed {len(PREVIEWS)} README previews and their provenance record")


if __name__ == "__main__":
    main()
