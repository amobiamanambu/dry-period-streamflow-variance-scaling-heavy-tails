#!/usr/bin/env python3
"""Stage 01: download annual GridMET minimum and maximum temperature files."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

from lib.common import (
    copy_provenance, load_config, parse_args, prepare_stage, configure_logging,
    write_receipt, year_range,
)


STAGE = 1


def download(url: str, destination: Path, retries: int, timeout: int, logger, force: bool) -> None:
    if destination.exists() and destination.stat().st_size > 100_000 and not force:
        logger.info("Existing file retained: %s", destination.name)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "continental-catchment-pipeline/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as output:
                expected = int(response.headers.get("Content-Length", 0))
                while block := response.read(1024 * 1024):
                    output.write(block)
            if partial.stat().st_size < 100_000:
                raise IOError(f"Downloaded file is implausibly small: {partial.stat().st_size} bytes")
            if expected and partial.stat().st_size != expected:
                raise IOError(f"Content-Length mismatch: {partial.stat().st_size} != {expected}")
            partial.replace(destination)
            logger.info("Downloaded %s (%.1f MB)", destination.name, destination.stat().st_size / 1e6)
            return
        except (OSError, urllib.error.URLError) as error:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url} after {retries} attempts") from error
            delay = 2 ** (attempt - 1)
            logger.warning("Download attempt %d failed for %s; retrying in %ds", attempt, url, delay)
            time.sleep(delay)


def main() -> None:
    args = parse_args(__doc__ or "", STAGE)
    cfg = load_config(args.config)
    out = prepare_stage(cfg, STAGE, args.force)
    logger = configure_logging(cfg, STAGE)
    copy_provenance(cfg)
    base_url = cfg["gridmet"]["base_url"].rstrip("/")
    outputs = []
    years = list(year_range(cfg))
    if args.limit:
        years = years[: args.limit]
    for variable, template in cfg["gridmet"]["variables"].items():
        for year in years:
            filename = template.format(year=year)
            destination = out / "netcdf" / variable / filename
            download(
                f"{base_url}/{filename}", destination,
                cfg["gridmet"]["download_retries"],
                cfg["gridmet"]["download_timeout_seconds"], logger, args.force,
            )
            outputs.append(destination)
    write_receipt(cfg, STAGE, outputs, {"files": len(outputs), "years": years})
    logger.info("Stage 01 complete: %d NetCDF files", len(outputs))


if __name__ == "__main__":
    main()
