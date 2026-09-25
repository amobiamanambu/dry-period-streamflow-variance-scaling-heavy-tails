#!/usr/bin/env python3
"""Audit the repository for scope, portability, and integrity."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "provenance" / "FILE_MANIFEST_SHA256.csv"
MAX_BYTES = 50 * 1024 * 1024
EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "outputs", "workspace", "__pycache__",
    ".pytest_cache", ".ruff_cache", "build", "dist",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files() -> list[Path]:
    """Return only files intended for Git, excluding generated/local trees."""
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        files.append(path)
    return sorted(files)


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []
    required = [
        "README.md", "CITATION.cff", "Makefile", "config/full_study.json",
        "data/example/example_transitions.csv.gz",
        "data/derived/basin_scaling.csv.gz",
        "data/derived/basin_distribution.csv.gz",
        "data/derived/basin_prediction_q10.csv.gz",
        "data/derived/basin_place_metrics.csv.gz",
        "data/expected/reported_claims.csv",
        "docs/figures/01_real_gage_estimator_check.png",
        "docs/figures/02_continental_result_audit.png",
        "docs/figures/PROVENANCE.json",
        "docs/CLAIM_MAP.md",
        "provenance/HISTORICAL_CODE_MAP.csv",
        "provenance/SOURCE_MAP.csv",
        "provenance/FILE_MANIFEST_SHA256.csv",
    ]
    for relative in required:
        if not (REPO_ROOT / relative).exists():
            failures.append(f"missing required file: {relative}")

    allowed_receipts = {
        "stage_08_matched_qc.json",
        "stage_10_transitions.json",
        "stage_14_recession.json",
        "stage_15_continental_results.json",
        "stage_16_scaling_tests.json",
        "stage_17_distribution_tests.json",
    }
    receipt_dir = REPO_ROOT / "provenance" / "receipts"
    if receipt_dir.exists():
        observed_receipts = {path.name for path in receipt_dir.glob("*.json")}
        for name in sorted(observed_receipts - allowed_receipts):
            failures.append(
                f"unregistered execution receipt: provenance/receipts/{name}"
            )
        for name in sorted(allowed_receipts - observed_receipts):
            failures.append(
                f"missing registered execution receipt: provenance/receipts/{name}"
            )
    else:
        failures.append("missing registered receipt directory: provenance/receipts")

    forbidden_names = {"manuscript", "wrr_figure_suite", "continental_run"}
    files = release_files()
    for path in files:
        relative = path.relative_to(REPO_ROOT)
        if any(part in forbidden_names for part in relative.parts):
            failures.append(f"forbidden release path: {relative}")
        if path.is_file() and path.stat().st_size > MAX_BYTES:
            failures.append(f"file exceeds 50 MiB: {relative}")
        if path.is_file() and path.suffix.lower() in {".docx", ".pdf", ".rdb", ".nc", ".sqlite"}:
            failures.append(f"excluded artifact type present: {relative}")

    sensitive_literals = (
        "/Users/" + "acamanambu",
        "/Users/" + "amobichukwu",
        "amobijones" + "@gmail.com",
    )
    sensitive_patterns = {
        "absolute POSIX user-home path": re.compile(
            r"(?<![A-Za-z0-9])/(?:Users|home)/[^/\s\"'<>]+/"
        ),
        "absolute Windows user-home path": re.compile(
            r"(?i)\b[A-Z]:\\Users\\[^\\\s\"'<>]+\\"
        ),
        "private key material": re.compile(
            "-----" + r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        ),
        "GitHub access token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        "API token with sk- prefix": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "AWS access key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    }
    text_suffixes = {".py", ".md", ".json", ".csv", ".yml", ".yaml", ".toml", ".txt", ""}
    for path in files:
        if path.suffix.lower() not in text_suffixes:
            continue
        if path.stat().st_size > 10 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for literal in sensitive_literals:
            if literal in text:
                failures.append(f"workstation identifier in {path.relative_to(REPO_ROOT)}: {literal}")
        for label, pattern in sensitive_patterns.items():
            if pattern.search(text):
                failures.append(
                    f"{label} in {path.relative_to(REPO_ROOT)}"
                )

    code_map_path = REPO_ROOT / "provenance" / "HISTORICAL_CODE_MAP.csv"
    recorded_hashes: set[str] = set()
    if code_map_path.exists():
        code_map = pd.read_csv(code_map_path, dtype=str).fillna("")
        expected_columns = [
            "component", "receipt_recorded_sha256", "archived_sha256",
            "release_sha256", "status", "note",
        ]
        if list(code_map.columns) != expected_columns:
            failures.append("historical code map has an unexpected schema")
        else:
            for row in code_map.itertuples(index=False):
                component = re.sub(r" \(Stages [^)]+\)$", "", row.component)
                archived = REPO_ROOT / "workflow" / "historical" / component
                release = REPO_ROOT / "workflow" / component
                if not archived.is_file():
                    failures.append(f"historical code-map target missing: {component}")
                elif sha256(archived) != row.archived_sha256:
                    failures.append(f"historical code-map hash differs: {component}")
                if not release.is_file():
                    failures.append(f"release code-map target missing: {component}")
                elif sha256(release) != row.release_sha256:
                    failures.append(f"release code-map hash differs: {component}")
                if re.fullmatch(r"[0-9a-f]{64}", row.receipt_recorded_sha256):
                    recorded_hashes.add(row.receipt_recorded_sha256)
                elif row.receipt_recorded_sha256 != "not recorded":
                    failures.append(f"invalid recorded code hash: {row.component}")
                if row.status == "exact" and not (
                    row.receipt_recorded_sha256
                    == row.archived_sha256
                    == row.release_sha256
                ):
                    failures.append(f"exact code-map row is not byte-identical: {row.component}")

    source_map_path = REPO_ROOT / "provenance" / "SOURCE_MAP.csv"
    if source_map_path.exists():
        source_map = pd.read_csv(source_map_path, dtype=str).fillna("")
        expected_columns = [
            "release_product", "frozen_upstream_reference",
            "frozen_producer_reference", "normalized_reconstruction_code",
            "release_assembly_code", "role",
        ]
        if list(source_map.columns) != expected_columns:
            failures.append("source map has an unexpected schema")
        else:
            for row in source_map.itertuples(index=False):
                for token in re.findall(r"#sha256=([^ +;]+)", row.frozen_producer_reference):
                    if not re.fullmatch(r"[0-9a-f]{64}", token):
                        failures.append(
                            f"invalid producer hash in source map: {row.release_product}"
                        )
                    elif token not in recorded_hashes:
                        failures.append(
                            f"unregistered producer hash in source map: {row.release_product}"
                        )
                for field in (
                    row.normalized_reconstruction_code,
                    row.release_assembly_code,
                ):
                    for relative in (item.strip() for item in field.split(";")):
                        if relative == "not applicable":
                            continue
                        if not (REPO_ROOT / relative).is_file():
                            failures.append(
                                f"source-map code target missing: {relative}"
                            )

    if MANIFEST.exists():
        manifest = pd.read_csv(MANIFEST)
        listed = set(manifest["path"].astype(str))
        actual = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in files
            if path != MANIFEST
        }
        for relative in sorted(actual - listed):
            failures.append(f"release file is absent from manifest: {relative}")
        for relative in sorted(listed - actual):
            failures.append(f"manifest lists a non-release or missing file: {relative}")
        for row in manifest.itertuples(index=False):
            path = REPO_ROOT / row.path
            if not path.exists():
                failures.append(f"manifest target missing: {row.path}")
                continue
            if path.stat().st_size != int(row.bytes):
                failures.append(f"manifest byte count differs: {row.path}")
            if sha256(path) != row.sha256:
                failures.append(f"manifest hash differs: {row.path}")

    preview_provenance = REPO_ROOT / "docs" / "figures" / "PROVENANCE.json"
    if preview_provenance.exists():
        preview = json.loads(preview_provenance.read_text(encoding="utf-8"))
        for section in ("source_sha256", "preview_sha256"):
            for relative, recorded in preview.get(section, {}).items():
                path = REPO_ROOT / relative
                if not path.exists():
                    failures.append(f"preview provenance target missing: {relative}")
                elif sha256(path) != recorded:
                    failures.append(f"README preview provenance is stale: {relative}")

    if not (REPO_ROOT / "LICENSE").exists():
        warnings.append("the repository currently grants no software or derived-data license")
    if "doi" not in (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8").lower():
        warnings.append("no archival DOI is recorded")

    report = {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "warnings": warnings,
    }
    destination = REPO_ROOT / "outputs" / "release_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        for item in sorted(set(failures)):
            print(f"FAIL: {item}")
        raise SystemExit(f"Release audit failed with {len(set(failures))} issue(s)")
    print("Release audit passed.")
    for item in warnings:
        print(f"NOTICE: {item}")


if __name__ == "__main__":
    main()
