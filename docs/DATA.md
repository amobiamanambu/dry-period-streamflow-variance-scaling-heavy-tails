# Data sources and release policy

## Upstream public sources

| Source | Variables used | Study period | Release treatment |
|---|---|---|---|
| U.S. Geological Survey National Water Information System | Daily mean discharge, parameter 00060, statistic 00003 | 1980–2025 | Use a checksummed batch snapshot outside GitHub; a later download may differ from the archived input |
| GAGES-II | Basin polygons, gage coordinates, reference status, hydroclimatic and disturbance attributes | Published release | Stage the published release outside GitHub and cite it |
| gridMET | Daily precipitation, reference evapotranspiration, minimum temperature, maximum temperature | 1980–2025 | Temperature download is scripted; the historical basin precipitation/PET preparation requires an archived snapshot or a new tested wrapper |
| Aggregated ecoregion labels | Descriptive regional strata carried with the GAGES-II basin inventory | Published GAGES-II release | Included as labels in compact tables; no separate boundary download is required by the numerical workflow |

GAGES-II reference: Falcone, J. A. (2011), *GAGES-II: Geospatial Attributes of Gages for Evaluating Streamflow*, U.S. Geological Survey, [https://doi.org/10.3133/70046617](https://doi.org/10.3133/70046617).

USGS daily-values service documentation: [https://nwis.waterservices.usgs.gov/docs/dv-service/daily-values-service-details/](https://nwis.waterservices.usgs.gov/docs/dv-service/daily-values-service-details/)

The staged USGS snapshot must include its download manifest. Stage 05 rejects
missing, unlisted, truncated, or hash-mismatched `.rdb.gz` batches before any
records are written to the analysis database.

gridMET project page: [https://www.climatologylab.org/gridmet.html](https://www.climatologylab.org/gridmet.html)

## Included data layers

### `data/example/`

The example table contains complete analysis-ready records for nine real gages, one per aggregated ecoregion. It retains the variables needed to reproduce dry-screen selection, conditional variance fitting, and standardized-tail diagnostics. Gage identifiers are stored as eight-character strings.

### `data/derived/`

These are compact, frozen analysis products, not raw observations. They contain the minimum basin-level fields needed to recompute the article's central counts, distributions, theory comparison, and prediction-skill contrasts. They deliberately omit the hundreds of unused GAGES-II attributes and all local file paths.

### Excluded large data

The repository does not contain the national NetCDF collection, USGS RDB batches, basin climate CSVs, SQLite database, thousands of per-basin daily files, prediction ensembles, bootstrap arrays, or manuscript figure assets. Those products exceed sensible Git history and are not needed for the fast numerical audit.

The compact basin-level products are sufficient to reproduce every value in the registered claim table. They do not substitute for the much larger daily inputs needed to refit all 5,227 accepted basins from the beginning; those inputs belong in a versioned DOI archive rather than Git history.

## Snapshot caveat

The original USGS download was made in January 2026 and included provisional values. The original gridMET extraction also included dates that were within the provider's preliminary window. Upstream services may revise those values. Exact verification therefore uses the frozen derived tables and their hashes; a current raw-data reconstruction tests robustness to the present public records.

## Licensing

The upstream data retain their source terms. This repository currently grants
no separate license for original code or derived tables, so reuse requires
permission unless a later tagged version supplies explicit license files. No
repository license can replace the terms attached to third-party source data.
