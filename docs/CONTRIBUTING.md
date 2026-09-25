# Contributing

Changes to scientific estimators must include a test and a description of their effect on the registered claims. Do not replace frozen reference products silently. A change that alters a registered value requires a new release, regenerated provenance manifest, and documented scientific rationale.

Keep raw data, manuscript files, and generated outputs out of Git history. Preserve `GAGE_ID` as a string. Use project-relative paths and deterministic sort order before aggregation or resampling.

Install `requirements-dev.txt` and run `make lint`, `make test`, and `make
audit` before proposing a change. The lint target includes every normalized,
historical, maintainer, verification, and test script so undefined names in
rarely executed full-workflow branches are caught in continuous integration.
