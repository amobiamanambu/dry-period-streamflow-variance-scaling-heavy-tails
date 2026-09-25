# Full continental workflow

The numbered workflow is provided for auditability and full reconstruction. It is not required to check the article's reported numbers; use `make verify` for that purpose.

## Resource envelope

Allow approximately 150 GB of free disk space and one to two days on a four-core workstation, in addition to download time. The historical run produced about 72 GB under its working directory.

## Required external inputs

Stage the inputs using the paths in `config/full_study.json`:

```text
data/external/basin_inventory.csv
data/external/gagesii_basins/GAGE_II_Basins.{shp,shx,dbf,prj}
data/external/basin_precipitation.csv
data/external/basin_pet.csv
data/external/usgs_daily/raw_rdb_gz/
data/external/usgs_daily/manifest.csv
data/external/discharge_qc.csv
data/external/gagesii_attributes.zip
```

Stage 05 verifies every selected USGS gzip batch against the byte count and
SHA-256 digest recorded in `data/external/usgs_daily/manifest.csv`. The GAGES-II
gage-point layer used for paper cartography is not part of the numerical
workflow. Stage 00 verifies the required polygon sidecars before Stage 03 reads
the shapefile bundle.

The precipitation, PET, discharge-batch, inventory, and discharge-QC preparation steps preceded the numbered workflow in the historical analysis. Their original ad hoc scripts are intentionally not presented as a clean pipeline. Raw reconstruction therefore requires one of the following external input arrangements:

1. the exact analysis inputs in a checksummed archive; or
2. newly acquired inputs prepared with independently tested commands before the entire workflow is rerun under the pinned configuration.

The frozen derived tables support complete numerical verification of the selected claim registry.

## Run order

After installing `requirements-full.txt` and staging the inputs:

```bash
python reproducibility/run_full_workflow.py \
  --confirm-full-run \
  --include-sensitivity-analyses \
  --workers 4
```

The guarded runner executes Stages 00–18 in dependency order and then the seven versioned sensitivity analyses used by the study. New stage receipts contain deterministic SHA-256 fingerprints for every declared file or directory, and a stage is not marked successful if any declared output is missing. Before a downstream stage starts, the relevant receipt is checked against the active configuration, recorded code, predecessor receipt, and output contents. A forced rerun clears that stage's output directory before replacement, preventing stale basin or year files from surviving. Use `--from-stage` and `--to-stage` only after any external-input changes have also been verified. The sensitivity sequence can be run separately with `reproducibility/run_sensitivity_analyses.py`.

Stages 04 and 05 deliberately share one SQLite file: Stage 05 appends discharge tables to the climate database. The first Stage-05 run validates the pristine Stage-04 database fingerprint. Before a forced Stage-05 reload, the existing Stage-05 fingerprint must match the complete post-ingest database; the runner then validates the Stage-04 receipt chain and existence of the shared file. A legacy Stage-05 receipt without content fingerprints is rejected for in-place reuse. This is the only declared output intentionally mutated by its immediate downstream stage.

Sensitivity programs are retained in `workflow/sensitivity_*.py`. They are separate, versioned post-processing analyses and do not overwrite the numbered stages. Their runner validates each numbered stage that a selected analysis reads directly. Selecting the robust-scale, selection, or signed-tail attribute analysis also validates or runs the shared tail-direction prerequisite in canonical order.

After a successful normalized run, rebuild the compact release layer with:

```bash
python maintainer/prepare_release_data.py \
  --workspace . \
  --analysis-root workspace/full_run \
  --basin-inventory data/external/basin_inventory.csv \
  --sensitivity-root workspace/full_run/sensitivity_analyses
```

## Historical provenance limits

The historical run did not preserve every intermediate configuration snapshot, and several early manifests stored absolute paths from the original workstation. The release therefore uses one normalized configuration and sanitized receipts. Bitwise identity to every historical intermediate is not claimed until a fresh full run is completed and archived.
