#!/usr/bin/env python3
"""Refresh the SHA-256 manifest after versioned repository-file changes."""

from __future__ import annotations

from prepare_release_data import REPO_ROOT, build_manifest


def main() -> None:
    build_manifest()
    print(f"Updated {REPO_ROOT / 'provenance' / 'FILE_MANIFEST_SHA256.csv'}")


if __name__ == "__main__":
    main()
