"""Download 20 years (1995-2014) of daily precipitation for Germany only,
from CPC (NOAA PSL OPeNDAP) and CMIP6 MPI-ESM1-2-HR (ESGF OPeNDAP).

Only the Germany bounding box is fetched via OPeNDAP subsetting, so no
full global files are downloaded. Outputs are small NetCDF files saved to
data/demo/, at daily, monthly-mean, and pure-spatial-mean-time-series
resolutions.
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DEMO_DIR = BASE_DIR / "data" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)
LARGE_DATA_DIR = BASE_DIR / "largeData"
LARGE_DATA_DIR.mkdir(parents=True, exist_ok=True)

GERMANY_BBOX = [5.5, 15.5, 47.0, 55.5]  # min_lon, max_lon, min_lat, max_lat
YEARS = list(range(1995, 2015))  # 1995-2014 inclusive, 20 years

CMIP6_CHUNKS = [
    "19950101-19991231",
    "20000101-20041231",
    "20050101-20091231",
    "20100101-20141231",
]
CMIP6_BASE_URL = (
    "https://esgf-node.ornl.gov/thredds/dodsC/css03_data/CMIP6/CMIP/MPI-M/"
    "MPI-ESM1-2-HR/historical/r1i1p1f1/day/pr/gn/v20190710/"
    "pr_day_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_{chunk}.nc"
)

CPC_BASE_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/cpc_global_precip/precip.{year}.nc"


def slice_lat(da, min_lat, max_lat):
    if da["lat"].values[0] > da["lat"].values[-1]:
        return slice(max_lat, min_lat)
    return slice(min_lat, max_lat)


def download_cpc_germany():
    print("=== Downloading CPC Germany subset, 1995-2014 ===")
    min_lon, max_lon, min_lat, max_lat = GERMANY_BBOX
    yearly = []
    for year in YEARS:
        url = CPC_BASE_URL.format(year=year)
        print(f"  Fetching {year}...", end=" ", flush=True)
        ds = xr.open_dataset(url)
        da = ds["precip"]
        lat_sl = slice_lat(da, min_lat, max_lat)
        da_r = da.sel(lon=slice(min_lon, max_lon), lat=lat_sl).load().astype("float32")
        yearly.append(da_r)
        ds.close()
        print(f"OK ({dict(da_r.sizes)})")

    da_all = xr.concat(yearly, dim="time")
    ds_out = da_all.to_dataset(name="precip")
    ds_out.attrs.update(
        {
            "title": "CPC Global Unified Precipitation - Germany subset",
            "source": "NOAA PSL OPeNDAP (cpc_global_precip)",
            "region": "Germany",
            "bbox": str(GERMANY_BBOX),
            "period": "1995-2014",
        }
    )

    out_daily = LARGE_DATA_DIR / "cpc_germany_1995_2014_daily.nc"
    ds_out.to_netcdf(out_daily)
    print(f"  Saved daily: {out_daily} ({out_daily.stat().st_size / 1e6:.3f} MB)")

    monthly = ds_out.resample(time="MS").mean()
    out_monthly = DEMO_DIR / "cpc_germany_1995_2014_monthly.nc"
    monthly.to_netcdf(out_monthly)
    print(f"  Saved monthly: {out_monthly} ({out_monthly.stat().st_size / 1e6:.4f} MB)")


def download_cmip6_germany():
    print("\n=== Downloading CMIP6 Germany subset, 1995-2014 ===")
    min_lon, max_lon, min_lat, max_lat = GERMANY_BBOX
    lon_min_360 = min_lon % 360
    lon_max_360 = max_lon % 360

    chunks = []
    for chunk in CMIP6_CHUNKS:
        url = CMIP6_BASE_URL.format(chunk=chunk)
        print(f"  Fetching {chunk}...", end=" ", flush=True)
        ds = xr.open_dataset(url)
        pr = ds["pr"]
        lat_sl = slice_lat(pr, min_lat, max_lat)
        pr_r = pr.sel(lon=slice(lon_min_360, lon_max_360), lat=lat_sl).load().astype("float32")
        pr_r = pr_r * 86400.0  # kg m-2 s-1 -> mm/day
        pr_r.attrs["units"] = "mm/day"
        chunks.append(pr_r)
        ds.close()
        print(f"OK ({dict(pr_r.sizes)})")

    pr_all = xr.concat(chunks, dim="time")
    ds_out = pr_all.to_dataset(name="pr")
    ds_out.attrs.update(
        {
            "title": "MPI-ESM1-2-HR historical precipitation - Germany subset",
            "source": "ESGF OPeNDAP (CMIP6.CMIP.MPI-M.MPI-ESM1-2-HR.historical.r1i1p1f1.day.pr.gn)",
            "region": "Germany",
            "bbox": str(GERMANY_BBOX),
            "period": "1995-2014",
        }
    )

    out_daily = LARGE_DATA_DIR / "cmip6_germany_1995_2014_daily.nc"
    ds_out.to_netcdf(out_daily)
    print(f"  Saved daily: {out_daily} ({out_daily.stat().st_size / 1e6:.3f} MB)")

    monthly = ds_out.resample(time="MS").mean()
    out_monthly = DEMO_DIR / "cmip6_germany_1995_2014_monthly.nc"
    monthly.to_netcdf(out_monthly)
    print(f"  Saved monthly: {out_monthly} ({out_monthly.stat().st_size / 1e6:.4f} MB)")


if __name__ == "__main__":
    download_cpc_germany()
    download_cmip6_germany()
    print("\nDone.")
