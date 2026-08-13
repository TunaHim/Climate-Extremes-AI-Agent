#!/usr/bin/env python3
"""
Download ERA5 hourly precipitation from Copernicus Climate Data Store (CDS).

Prerequisites:
    1. Create a CDS account: https://cds.climate.copernicus.eu/
    2. Install the CDS API key as described in:
       https://cds.climate.copernicus.eu/how-to-api
    3. Install cdsapi: pip install cdsapi

Usage:
    python scripts/download_era5_precip.py
"""

from pathlib import Path

import cdsapi

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "era5"

# Geographic subset: Europe
AREA = [75, -15, 30, 45]  # North, West, South, East

# Must overlap with CMIP6 simulation period
TARGET_YEAR = 2013
TARGET_MONTH = "07"

YEARS = [str(TARGET_YEAR)]
MONTHS = [TARGET_MONTH]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00" for h in range(24)]


def download_era5_precip() -> Path:
    """Download ERA5 hourly total precipitation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"era5_total_precipitation_{TARGET_YEAR}{TARGET_MONTH}_europe.nc"

    client = cdsapi.Client()

    request = {
        "product_type": ["reanalysis"],
        "variable": ["total_precipitation"],
        "year": YEARS,
        "month": MONTHS,
        "day": DAYS,
        "time": TIMES,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }

    print(f"Requesting ERA5 precipitation for {YEARS}-{MONTHS} ...")
    client.retrieve(
        "reanalysis-era5-single-levels",
        request,
        str(out_file),
    )

    print(f"Saved to: {out_file}")
    return out_file


def main():
    download_era5_precip()
    print("Done.")


if __name__ == "__main__":
    main()
