"""Data loading helpers for GRIB and NetCDF files."""

from pathlib import Path

import xarray as xr


def open_grib_dataset(path: Path, **kwargs) -> xr.Dataset:
    """Open a GRIB file with cfgrib engine."""
    return xr.open_dataset(
        str(path),
        engine="cfgrib",
        decode_times=True,
        **kwargs,
    )


def open_netcdf_dataset(path: Path, **kwargs) -> xr.Dataset:
    """Open a NetCDF file."""
    return xr.open_dataset(str(path), **kwargs)


def load_era5_precip(path: Path) -> xr.DataArray:
    """Load ERA5 total precipitation and convert to mm/day."""
    ds = open_netcdf_dataset(path)
    # ERA5 total precipitation is in meters of water (per hour)
    # Convert to mm/day by summing over hours and multiplying by 1000
    da = ds["tp"]
    da = da * 1000.0  # m -> mm
    da.attrs["units"] = "mm"
    return da


def load_destine_precip(path: Path) -> xr.DataArray:
    """Load DestinE total precipitation and convert to mm/day."""
    ds = open_grib_dataset(path)
    # Param 228 is total precipitation in meters
    da = ds["tp"]
    da = da * 1000.0  # m -> mm
    da.attrs["units"] = "mm"
    return da


def load_cmip6_precip(path: Path, var: str = "pr") -> xr.DataArray:
    """Load CMIP6 precipitation (kg m-2 s-1) and convert to mm/day."""
    ds = open_netcdf_dataset(path)
    da = ds[var]
    # pr is precipitation flux in kg m-2 s-1 == mm s-1
    da = da * 86400.0  # mm/s -> mm/day
    da.attrs["units"] = "mm/day"
    return da


def load_cpc_precip(path: Path, var: str = "precip") -> xr.DataArray:
    """Load CPC daily precipitation (already in mm/day)."""
    ds = open_netcdf_dataset(path)
    da = ds[var]
    da.attrs["units"] = "mm/day"
    return da
