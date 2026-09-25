# Historical and normalized source lineage

This directory preserves historical analysis structure, receipt-matched source
where byte identity survives, normalized supplementary-analysis implementations,
and the Stage 05 ingest script that preceded manifest enforcement.
`provenance/HISTORICAL_CODE_MAP.csv` distinguishes these cases explicitly.
The files are retained as provenance evidence and are not the default execution
path.

The corresponding files one level above are the release runners. Where necessary, those copies use configuration-driven paths and deterministic ordering so a new run is portable and independent of worker completion order. The numerical products in `data/derived/` remain tied to the historical hashes until the normalized workflow is run again and archived.

`provenance/HISTORICAL_CODE_MAP.csv` records the relationship explicitly. One
limitation is retained: the earlier `lib/common.py` snapshot recorded by the
Stage 14–16 receipts is no longer available. The archived helper is the later
snapshot used by Stages 17–18. The code map records exactness row by row.
Receipt-recorded hashes survive for source files marked `exact` and for the
historical copies of several ported runners. Stages 15 and 18 and the
supplementary-analysis copies use normalized public wording or output records;
the map records their archived execution hashes and current public hashes
without claiming byte identity.
