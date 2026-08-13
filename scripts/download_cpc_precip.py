#!/usr/bin/env python3
"""
Download CPC Unified Gauge-Based Global Daily Precipitation as a reference dataset.

Source: NOAA PSL
URL pattern: https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/YYYY/YYYYMM.nc

Usage:
    python scripts/download_cpc_precip.py
"""

from pathlib import Path

import requests
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "cpc"

# Time period to download (must overlap with simulation)
TARGET_YEAR = 2013
TARGET_MONTH = "07"

BASE_URL = "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip"


def download_year(year: int) -> Path:
    """Download one year of CPC daily precipitation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/precip.{year}.nc"
    out_file = OUTPUT_DIR / f"cpc_precip_{year}.nc"

    if out_file.exists():
        print(f"Already exists: {out_file}")
        return out_file

    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(out_file, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"Saved to: {out_file}")
    return out_file


def main():
    download_year(TARGET_YEAR)
    print("Done.")


if __name__ == "__main__":
    main()
