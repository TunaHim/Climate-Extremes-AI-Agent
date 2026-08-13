#!/usr/bin/env python3
"""Create lightweight demo datasets for Streamlit Cloud deployment.

Produces:
- data/demo/cpc_south_asia_2013.nc     : daily CPC subset for agent extremes
- data/demo/cpc_europe_2013.nc         : daily CPC subset for agent extremes
- data/demo/cmip6_global_monthly_2013.nc : CMIP6 monthly means (pr -> mm/day)
- data/demo/cpc_global_monthly_2013_regridded.nc : CPC monthly means regridded to CMIP6 grid
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

OUT_DIR = BASE_DIR / "data" / "demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CPC_FULL = BASE_DIR / "data" / "cpc" / "cpc_precip_2013.nc"
CMIP6_FULL = (
    BASE_DIR
    / "data"
    / "cmip6"
    / "pr_day_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_20100101-20141231.nc"
)


def _standardise_coords(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    for old, new in [("latitude", "lat"), ("longitude", "lon")]:
        if old in da.coords and new not in da.coords:
            rename[old] = new
    if rename:
        da = da.rename(rename)
    return da


def _to_netcdf(ds: xr.Dataset, path: Path) -> None:
    ds.to_netcdf(path, engine="netcdf4")
    size_mb = path.stat().st_size / 1e6
    dims = dict(ds.dims)
    print(f"  created {path.name}: {size_mb:.2f} MB, dims={dims}")


def make_cpc_daily_subset(ds: xr.Dataset, bbox: list[float]) -> xr.Dataset:
    """Create a regional daily CPC subset."""
    min_lon, max_lon, min_lat, max_lat = bbox
    da = _standardise_coords(ds["precip"])
    if da["lat"].values[0] > da["lat"].values[-1]:
        lat_slice = slice(max_lat, min_lat)
    else:
        lat_slice = slice(min_lat, max_lat)
    da_region = da.sel(lon=slice(min_lon, max_lon), lat=lat_slice)
    ds_region = da_region.to_dataset(name="precip")
    ds_region.attrs.update(ds.attrs)
    return ds_region


def make_cpc_monthly_regridded(ds_cpc: xr.Dataset, target_grid: xr.DataArray) -> xr.Dataset:
    """Compute CPC monthly means and regrid to the CMIP6 grid."""
    da = _standardise_coords(ds_cpc["precip"])
    monthly = da.resample(time="1ME").mean(dim="time")
    target = _standardise_coords(target_grid)
    regridded = monthly.interp(
        lat=target["lat"],
        lon=target["lon"],
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    # Downcast to float32 to keep file size comparable to CMIP6 output
    return regridded.astype("float32").to_dataset(name="precip")


def make_cmip6_monthly(path: Path) -> xr.Dataset:
    """Select 2013, convert pr to mm/day, and compute monthly means."""
    with xr.open_dataset(path) as ds:
        pr_mm_day = ds["pr"] * 86400.0
        pr_2013 = pr_mm_day.sel(time=pr_mm_day.time.dt.year == 2013)
        monthly = pr_2013.resample(time="1ME").mean(dim="time")
        ds_out = monthly.astype("float32").to_dataset(name="pr")
        ds_out.attrs.update(ds.attrs)
        ds_out["pr"].attrs["units"] = "mm/day"
    return ds_out


def main():
    print("Creating demo datasets in", OUT_DIR)

    print("\nCPC daily regional subsets:")
    with xr.open_dataset(CPC_FULL) as ds_cpc:
        regions = {
            "cpc_south_asia_2013.nc": [60, 100, 5, 35],
            "cpc_europe_2013.nc": [-15, 45, 35, 75],
        }
        for fname, bbox in regions.items():
            ds_region = make_cpc_daily_subset(ds_cpc, bbox)
            _to_netcdf(ds_region, OUT_DIR / fname)

    print("\nCMIP6 global monthly means:")
    ds_cmip6_monthly = make_cmip6_monthly(CMIP6_FULL)
    _to_netcdf(ds_cmip6_monthly, OUT_DIR / "cmip6_global_monthly_2013.nc")

    target_grid = xr.DataArray(
        np.zeros((len(ds_cmip6_monthly["lat"]), len(ds_cmip6_monthly["lon"]))),
        dims=("latitude", "longitude"),
        coords={
            "latitude": ds_cmip6_monthly["lat"].values,
            "longitude": ds_cmip6_monthly["lon"].values,
        },
    )

    print("\nCPC global monthly means (regridded to CMIP6 grid):")
    with xr.open_dataset(CPC_FULL) as ds_cpc:
        ds_cpc_monthly = make_cpc_monthly_regridded(ds_cpc, target_grid)
        _to_netcdf(
            ds_cpc_monthly,
            OUT_DIR / "cpc_global_monthly_2013_regridded.nc",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
