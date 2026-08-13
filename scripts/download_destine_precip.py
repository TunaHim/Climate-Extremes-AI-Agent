#!/usr/bin/env python3
"""
Download IFS-FESOM precipitation data from DestinE Harmonised Data Access (HDA).

Usage:
    python scripts/download_destine_precip.py

You will be prompted for your DestinE Service Platform (DESP) credentials.
"""

import os
import re
import time
from getpass import getpass
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

import destinelab as deauth


# Load credentials from a .env file in the project root (optional)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


BASE_URL = "https://hda.data.destination-earth.eu"
SEARCH_URL = f"{BASE_URL}/stac/search"
COLLECTION = "EO.ECMWF.DAT.DT_CLIMATE_ADAPTATION"

# Configuration: small first test over Europe, July 2023
CONFIG = {
    "class": "d1",
    "dataset": "climate-dt",
    "activity": "ScenarioMIP",
    "experiment": "SSP3-7.0",
    "model": "IFS-FESOM",  # or "IFS-NEMO" or "ICON"
    "generation": "1",
    "realization": "1",
    "resolution": "standard",  # "standard" = H128, "high" = H1024
    "expver": "0001",
    "stream": "clte",  # high-frequency climate fields
    "time": "0000",
    "type": "fc",
    "levtype": "sfc",
    "param": "228",  # Total precipitation (tp) in meters
}

DATECHOICE = "2023-07-01T00:00:00Z"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "destine"


def authenticate() -> tuple[str, dict]:
    """Authenticate with DESP and return access token + headers."""
    username = os.getenv("DESP_USERNAME") or input("DESP username/email: ")
    password = os.getenv("DESP_PASSWORD") or getpass("DESP password: ")

    auth = deauth.AuthHandler(username, password)
    token = auth.get_token()
    if token is None:
        raise RuntimeError("Failed to obtain access token. Check credentials.")

    print("Access token obtained successfully.")
    headers = {"Authorization": f"Bearer {token}"}
    return token, headers


def check_dt_access(auth, token: str) -> None:
    """Check that Climate DT output access is granted."""
    try:
        auth.is_DTaccess_allowed(token)
        print("Climate DT access granted.")
    except Exception as exc:
        print(f"Warning: could not verify DT access: {exc}")


def build_filters(config: dict) -> dict:
    """Convert flat config into HDA query filters."""
    return {key: {"eq": value} for key, value in config.items()}


def submit_request(headers: dict, datechoice: str, filters: dict) -> dict:
    """Submit STAC search request and return the first product."""
    retry_strategy = Retry(
        total=5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        backoff_factor=1,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)

    print(f"Searching collection {COLLECTION} for {datechoice} ...")
    response = session.post(
        SEARCH_URL,
        headers=headers,
        json={
            "collections": [COLLECTION],
            "datetime": datechoice,
            "query": filters,
        },
        timeout=120,
    )

    if response.status_code != 200:
        print(response.text)
        response.raise_for_status()

    features = response.json().get("features", [])
    if not features:
        raise RuntimeError("No products found for the requested configuration.")

    product = features[0]
    print(f"Product found: {product['id']}")
    return product


def download_product(session: requests.Session, product: dict, headers: dict, out_dir: Path) -> Path:
    """Poll and download the product."""
    out_dir.mkdir(parents=True, exist_ok=True)
    download_url = product["assets"]["downloadLink"]["href"]

    print(f"Requesting download ...")
    response = session.get(download_url, headers=headers, stream=True, timeout=120)

    HTTP_SUCCESS = 200
    HTTP_ACCEPTED = 202

    # Poll while queued
    while response.status_code == HTTP_ACCEPTED or "Content-Disposition" not in response.headers:
        status = response.json().get("status", "queued") if response.headers.get("content-type", "").startswith("application/json") else "queued"
        print(f"  Order status: {status}")
        time.sleep(5)
        location = response.headers.get("Location", download_url)
        response = session.get(location, headers=headers, stream=True, timeout=120)

    if response.status_code != HTTP_SUCCESS:
        print(response.text)
        response.raise_for_status()

    if "Content-Disposition" not in response.headers:
        raise RuntimeError("Response missing Content-Disposition header.")

    filename = re.findall(r'filename="?(.+)"?', response.headers["Content-Disposition"])[0]
    out_path = out_dir / filename
    total_size = int(response.headers.get("content-length", 0))

    print(f"Downloading {filename} ({total_size / 1e6:.1f} MB) ...")
    with open(out_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"Saved to: {out_path}")
    return out_path


def main():
    token, headers = authenticate()

    auth = deauth.AuthHandler("", "")
    check_dt_access(auth, token)

    filters = build_filters(CONFIG)
    product = submit_request(headers, DATECHOICE, filters)

    retry_strategy = Retry(
        total=5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        backoff_factor=1,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)

    download_product(session, product, headers, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
