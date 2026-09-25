# Reproducibility model

## What each command establishes

`make verify` recomputes 81 selected numerical and inferential claims from the frozen, basin-level products in `data/derived/`. The registry covers headline sample counts, continental distributions, theory comparisons, prediction-skill contrasts under both bootstrap schemes, and the descriptive HUC2 comparisons; it is not a registry of every number printed in the article. The command also writes a complete HUC2 summary table to `outputs/verification/huc02_summary.csv`. Manuscript locators are listed in `CLAIM_MAP.md`.

`make demo` first reconstructs the example-gage selection and then applies the same conditional-moment and innovation estimators to complete records from the nine selected gages. Within each aggregated ecoregion, eligibility required Tier 1 status, at least 1,000 primary transitions, an estimable exponent, and an archived three-day Q10 evaluation. Selection minimized robust multivariate distance from the regional center across six prespecified diagnostics. The executable selection rule is in `reproducibility/example_selection.py`, and its output is `outputs/example/example_gage_selection.csv`. These are transparent examples rather than an independent validation sample.

`make full` prints the guarded command for the national workflow in `workflow/`. Running that command requires separately acquired public inputs and substantial storage and compute. The original raw snapshots included provisional observations, so a future redownload may not be byte-identical even when the analysis code is unchanged.

## What “reproduce” means here

The release supports numerical reproduction of the selected registered summaries and computational reproduction of the core estimator. Independent reconstruction from mutable upstream services is documented, but it is not represented as byte-for-byte replication of the archived 2026 input snapshots.

This distinction matters because USGS daily values may be revised and recent gridMET fields may be preliminary. The frozen derived products and SHA-256 manifest provide the stable evidentiary layer for the article.

The dry-period screen treats a missing precipitation or temperature value as unknown forcing. Any transition window containing such a day is excluded; missing forcing is never interpreted as evidence of zero precipitation or zero snowmelt risk.

## Scientific boundaries

- The exponent is estimated for finite daily increments, not an infinitesimal diffusion coefficient.
- A center near `m = 2` does not imply that all basins share one exponent.
- Statistical irreversibility is a path-asymmetry measure and is not physical entropy production.
- Low-flow skill is conditional on observed continuation of the screened dry period.
- Ecoregions and HUC2 units are descriptive reporting strata, not causal process domains.

## Exactness and tolerances

Integer counts must agree exactly. Floating-point claims are compared with explicit absolute tolerances in `data/expected/reported_claims.csv`. The tolerances reflect reported rounding, not permission for materially different results.
