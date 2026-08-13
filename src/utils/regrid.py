"""Regridding utilities for aligning model and reference grids."""

from pathlib import Path

import numpy as np
import xarray as xr


def create_target_grid(resolution: float = 0.5) -> xr.DataArray:
    """Create a regular lat/lon target grid (0-360 lon, -90-90 lat)."""
    lon = np.arange(resolution / 2, 360, resolution)
    lat = np.arange(-90 + resolution / 2, 90, resolution)
    return xr.DataArray(
        np.zeros((len(lat), len(lon))),
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )


def _standardise_coords(da: xr.DataArray) -> xr.DataArray:
    """Rename latitude/longitude coords to lat/lon for xarray interp."""
    rename = {}
    if "latitude" in da.coords and "lat" not in da.coords:
        rename["latitude"] = "lat"
    if "longitude" in da.coords and "lon" not in da.coords:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    return da


def regrid_to_common(
    da: xr.DataArray,
    target: xr.DataArray,
    method: str = "bilinear",
    reuse_weights: bool = True,
) -> xr.DataArray:
    """Regrid a DataArray to a target grid."""
    da = _standardise_coords(da)

    # Prefer xesmf for accurate conservative regridding; fall back to xarray.interp
    try:
        import xesmf as xe
        regridder = xe.Regridder(
            da,
            target,
            method=method,
            reuse_weights=reuse_weights,
            ignore_degenerate=True,
        )
        return regridder(da)
    except ImportError:
        pass

    # Fallback: xarray.interp works for any rectilinear grid
    target = _standardise_coords(target)
    new_lon = target["lon"]
    new_lat = target["lat"]
    # xarray.interp uses "linear" for 2D, while xesmf uses "bilinear"
    interp_method = "linear" if method == "bilinear" else method
    return da.interp(
        lon=new_lon,
        lat=new_lat,
        method=interp_method,
        kwargs={"fill_value": "extrapolate"},
    )


def regrid_dataset(
    ds: xr.Dataset,
    resolution: float = 0.5,
    method: str = "bilinear",
) -> xr.Dataset:
    """Regrid all data variables in a dataset to a common regular grid."""
    target = create_target_grid(resolution=resolution)
    regridded = {}
    for var in ds.data_vars:
        regridded[var] = regrid_to_common(ds[var], target, method=method)
    return xr.Dataset(regridded)
