#!/usr/bin/env python3
"""
Download CMIP6 daily precipitation data from ESGF.

Usage:
    python scripts/download_cmip6_precip.py

Example configuration below downloads MPI-ESM1-2-HR historical daily precipitation.
"""

import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "cmip6"

# Search configuration
SEARCH_NODE = "https://esgf-node.llnl.gov"
SEARCH_URL = f"{SEARCH_NODE}/esg-search/search/"

# Time period to download (must overlap with ERA5 reference)
TARGET_YEAR = 2013
TARGET_MONTH = "07"

CONFIG = {
    "type": "File",
    "project": "CMIP6",
    "variable_id": "pr",
    "source_id": "MPI-ESM1-2-HR",
    "experiment_id": "historical",
    "table_id": "day",
    "member_id": "r1i1p1f1",
    "format": "application/solr+json",
    "limit": 50,
    "offset": 0,
}


def search_files(config: dict) -> list[dict]:
    """Query ESGF search API and return file records."""
    print(f"Searching ESGF at {SEARCH_NODE} ...")
    response = requests.get(SEARCH_URL, params=config, timeout=120)
    response.raise_for_status()
    data = response.json()
    docs = data.get("response", {}).get("docs", [])
    print(f"Found {len(docs)} files")
    return docs


def pick_download_url(file_record: dict) -> str | None:
    """Pick the first HTTP download URL from file record."""
    urls = file_record.get("url", [])
    for url_entry in urls:
        # url entries look like: "http://...|HTTPServer|..."
        parts = url_entry.split("|")
        if len(parts) >= 2 and "http" in parts[0].lower():
            return parts[0]
    return None


def download_file(url: str, out_path: Path) -> Path:
    """Download a file with progress bar."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    print(f"Downloading {out_path.name} ({total_size / 1e6:.1f} MB) ...")
    with open(out_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print(f"Saved to: {out_path}")
    return out_path


def _file_covers_period(title: str, year: int) -> bool:
    """Check if a CMIP6 filename covers the requested year."""
    match = re.search(r"(\d{8})-(\d{8})\.nc", title)
    if not match:
        return False
    start_year = int(match.group(1)[:4])
    end_year = int(match.group(2)[:4])
    return start_year <= year <= end_year


def main():
    records = search_files(CONFIG)
    if not records:
        print("No CMIP6 files found for the requested configuration.")
        print("Try changing source_id, experiment_id, or variable_id.")
        return

    # Filter records to the requested year to limit download size
    filtered = [
        r for r in records
        if _file_covers_period(r.get("title", ""), TARGET_YEAR)
    ]
    print(f"Filtered to {len(filtered)} files covering year {TARGET_YEAR}")

    if not filtered:
        print("No files matched the target period. Showing available files:")
        for r in records[:5]:
            print(" -", r.get("title", "unknown"))
        return

    downloaded = []
    for record in filtered[:3]:  # download up to 3 matching chunks
        url = pick_download_url(record)
        if not url:
            continue
        filename = Path(urlparse(url).path).name
        out_path = OUTPUT_DIR / filename
        if out_path.exists():
            print(f"Already exists: {out_path}")
            downloaded.append(out_path)
            continue
        try:
            download_file(url, out_path)
            downloaded.append(out_path)
        except Exception as exc:
            print(f"Failed to download {url}: {exc}")
        time.sleep(1)  # be polite to ESGF nodes

    print(f"Downloaded {len(downloaded)} files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
