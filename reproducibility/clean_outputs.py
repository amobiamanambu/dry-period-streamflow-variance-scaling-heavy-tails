#!/usr/bin/env python3
"""Remove only generated reproduction outputs."""

import argparse
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "outputs"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    for name in ("example", "verification", "figures", ".matplotlib"):
        target = OUTPUT / name
        if target.exists():
            shutil.rmtree(target)
    audit = OUTPUT / "release_audit.json"
    if audit.exists():
        audit.unlink()
    print("Removed generated files under outputs/.")


if __name__ == "__main__":
    main()
