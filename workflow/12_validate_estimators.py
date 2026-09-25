#!/usr/bin/env python3
"""Stage 12: validate irreversibility and conditional recession-variance estimators."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lib.analysis import analyze_irreversibility, analyze_recession
from lib.common import (
    load_config, parse_args, prepare_stage, configure_logging, require_stage,
    stage_dir, write_receipt,
)


STAGE = 12


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    require_stage(cfg, 11)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    selected = pd.read_csv(stage_dir(cfg, 11) / "validation_basins.csv", dtype={"GAGE_ID": str})
    if args.limit:
        selected = selected.head(args.limit)
    spectra_all, recession_all, bins_all, diagnostics, failures = [], [], [], [], []
    for number, row in enumerate(selected.itertuples(index=False), start=1):
        gage = str(row.GAGE_ID).zfill(8)
        try:
            frame = pd.read_csv(row.file, parse_dates=["date"], low_memory=False)
            rng = np.random.default_rng(cfg["project"]["random_seed"] + int(gage))
            spectra, irreversible = analyze_irreversibility(
                frame, cfg, rng, cfg["irreversibility"]["iaaft_surrogates_validation"],
                surrogates_all_bins=True,
            )
            recession, bins, recession_diagnostic = analyze_recession(
                frame, cfg, rng, cfg["recession"]["bootstrap_replicates_validation"],
                bootstrap_primary_only=False,
            )
            spectra.insert(0, "GAGE_ID", gage)
            recession.insert(0, "GAGE_ID", gage)
            if not bins.empty:
                bins.insert(0, "GAGE_ID", gage)
                bins_all.append(bins)
            spectra_all.append(spectra)
            recession_all.append(recession)
            diagnostics.append({"GAGE_ID": gage, **irreversible, **recession_diagnostic})
        except Exception as error:
            failures.append({"GAGE_ID": gage, "error": repr(error)})
            logger.exception("Validation failed for %s", gage)
        logger.info("Validated %d/%d basins", number, len(selected))
    spectra_frame = pd.concat(spectra_all, ignore_index=True) if spectra_all else pd.DataFrame()
    recession_frame = pd.concat(recession_all, ignore_index=True) if recession_all else pd.DataFrame()
    bins_frame = pd.concat(bins_all, ignore_index=True) if bins_all else pd.DataFrame()
    paths = {
        "spectra": out / "validation_irreversibility_spectra.csv",
        "recession": out / "validation_recession_summary.csv",
        "bins": out / "validation_recession_bins.csv",
        "diagnostics": out / "validation_diagnostics.csv",
        "failures": out / "validation_failures.csv",
    }
    spectra_frame.to_csv(paths["spectra"], index=False)
    recession_frame.to_csv(paths["recession"], index=False)
    bins_frame.to_csv(paths["bins"], index=False)
    pd.DataFrame(diagnostics).to_csv(paths["diagnostics"], index=False)
    pd.DataFrame(failures, columns=["GAGE_ID", "error"]).to_csv(paths["failures"], index=False)
    if not diagnostics:
        raise RuntimeError(f"No validation basin succeeded; inspect {paths['failures']}")
    report = out / "VALIDATION_DECISION.md"
    report.write_text(
        "# Estimator validation decision\n\n"
        f"Successful basins: {len(diagnostics)} / {len(selected)}.\n\n"
        "Before continuing, inspect bin sensitivity, IAAFT p-values, variance-exponent stability across "
        "tau=1/2/3 days, Chapman–Kolmogorov error, and finite-time local-Gaussian diagnostics. "
        "A missing diagnostic is recorded as unavailable and can never count as a pass. Inspect "
        "and differences between the "
        "Q-only, precipitation-screened monotone, and precipitation-screened dry-state samples. "
        "Stage 13 does not convert statistical irreversibility into physical entropy production, "
        "and the finite-time conditional variance is not claimed to identify a formal diffusion coefficient. "
        "This stage validates the estimators used for the continental physical descriptors; the later "
        "2016–2025 holdout in Stage 18 separately validates low-flow predictability horizons.\n",
        encoding="utf-8",
    )
    if failures:
        logger.warning("%d validation basins failed; see %s", len(failures), paths["failures"])
    metrics = {"selected": len(selected), "successful": len(diagnostics), "failed": len(failures)}
    write_receipt(cfg, STAGE, [*paths.values(), report], metrics)
    logger.info("Stage 12 complete; inspect VALIDATION_DECISION.md before the continental stages")


if __name__ == "__main__":
    main()
