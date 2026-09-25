# Provenance records

`SOURCE_MAP.csv` separates frozen-product lineage from normalized
reconstruction code. `frozen_upstream_reference` identifies the object from
which a mapped product was assembled. A reference beginning
`archived-output://` is a stable public alias for a frozen object; it is an
identifier, not a repository path or a claim that the object was created under
that normalized name. These aliases are descriptive and do not identify the
upstream bytes; the repository manifest protects the released compact
products. `frozen_producer_reference` is either a receipt-matched
repository file or a `recorded-execution://…#sha256=…` identifier. The latter
records the producer hash without claiming that an exact source file is
included. `normalized_reconstruction_code` names the current implementation for
a new run and is not asserted to have produced the frozen object.
`release_assembly_code` names code that selected, filtered, or copied the frozen
object into this compact repository.

`receipts/` contains structurally faithful, path-sanitized copies of the Stage
08, 10, and 14–17 execution receipts. Sanitization may replace an absolute
workstation prefix, but it does not rename analyses, metric fields, declared
outputs, or code entries. The Stage 08 and Stage 10 receipts did not record
code hashes, so they establish execution metadata and output lineage but not
exact source identity. The Stage 14–16 receipts record an earlier
`lib/common.py` hash whose source snapshot is unavailable; the other listed
Stage 14–17 producer hashes are represented as described in
`HISTORICAL_CODE_MAP.csv`.

No Stage 18 or supplementary-sensitivity receipt is included. Their current
public implementations use normalized names and output locations that differ
from the frozen executions. Rewriting historical receipts with those current
names would misstate the recorded execution. Their recorded producer hashes
remain explicit in `HISTORICAL_CODE_MAP.csv` and `SOURCE_MAP.csv`; current
`workflow/` files are reconstruction implementations, not byte-identical
producer evidence.

In `HISTORICAL_CODE_MAP.csv`, `receipt_recorded_sha256` identifies the recorded
execution. `archived_sha256` is the hash of the file currently stored under
`workflow/historical/`; equality with `receipt_recorded_sha256` establishes
that file's receipt identity. The status also describes whether the default
release implementation is exact, ported, normalized, or subject to a
historical limitation. `release_sha256` identifies the current default
workflow implementation. The
release hydrology helper applies a conservative missing-forcing rule: an
unavailable temperature or precipitation value excludes the affected
transition window instead of being interpreted as no snowmelt risk.

`FILE_MANIFEST_SHA256.csv` records every versioned repository file except the
manifest itself. Regenerate it after changing a versioned file:

```bash
make manifest
```

`maintainer/prepare_release_data.py` packages outputs from the completed
analysis root supplied with `--analysis-root`. Its optional
`--sensitivity-root` identifies the corresponding supplementary-analysis
output directory; by default that directory is `sensitivity_analyses` beneath
the supplied analysis root.
