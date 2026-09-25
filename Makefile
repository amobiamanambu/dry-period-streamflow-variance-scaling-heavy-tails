PYTHON ?= python3

.PHONY: help demo verify figures test lint manifest audit reproduce full clean

help:
	@echo "demo       Rerun core estimators on nine real gages"
	@echo "verify     Recompute registered continental results"
	@echo "figures    Build diagnostic provenance figures"
	@echo "test       Run automated tests"
	@echo "lint       Check every Python file for syntax and undefined names"
	@echo "manifest   Refresh hashes after versioned file changes"
	@echo "audit      Validate release contents and checksums"
	@echo "reproduce  Run demo, verification, figures, tests, and audit"
	@echo "full       Show the guarded full-workflow command"

demo:
	$(PYTHON) reproducibility/run_example.py

verify:
	$(PYTHON) reproducibility/verify_reported_results.py

figures: demo verify
	$(PYTHON) reproducibility/make_provenance_figures.py

test:
	$(PYTHON) -m unittest discover -s tests -v

lint:
	$(PYTHON) -m ruff check --select E9,F63,F7,F82 .

manifest:
	$(PYTHON) maintainer/update_manifest.py

audit:
	$(PYTHON) reproducibility/audit_release.py

reproduce: demo verify figures test audit

full:
	@echo "Read docs/FULL_WORKFLOW.md, stage the external inputs, then run:"
	@echo "python reproducibility/run_full_workflow.py --confirm-full-run --include-sensitivity-analyses --workers N"

clean:
	$(PYTHON) reproducibility/clean_outputs.py
