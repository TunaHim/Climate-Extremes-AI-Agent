"""Precipitation extreme indices."""

import numpy as np
import xarray as xr


def rx1day(da: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    """Maximum 1-day precipitation amount (RX1day)."""
    return da.resample({time_dim: "1YE"}).max(dim=time_dim, skipna=True).squeeze()


def rx5day(da: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    """Maximum 5-day consecutive precipitation amount (RX5day)."""
    rolling = da.rolling({time_dim: 5}, min_periods=1).sum()
    return rolling.resample({time_dim: "1YE"}).max(dim=time_dim, skipna=True).squeeze()


def r95p(da: xr.DataArray, wet_threshold: float = 1.0, time_dim: str = "time") -> xr.DataArray:
    """Total precipitation from days > 95th percentile of wet days."""
    wet_days = da.where(da >= wet_threshold)
    # 95th percentile over all wet days in the period
    p95 = wet_days.quantile(0.95, dim=time_dim, skipna=True)
    extreme_days = da.where(da > p95)
    return extreme_days.sum(dim=time_dim, skipna=True)


def prcptot(da: xr.DataArray, wet_threshold: float = 1.0, time_dim: str = "time") -> xr.DataArray:
    """Total precipitation from wet days (>= 1 mm/day)."""
    wet = da.where(da >= wet_threshold)
    return wet.sum(dim=time_dim, skipna=True)


def compute_extreme_indices(
    da: xr.DataArray,
    time_dim: str = "time",
    wet_threshold: float = 1.0,
) -> xr.Dataset:
    """Compute a suite of precipitation extreme indices."""
    return xr.Dataset(
        {
            "RX1day": rx1day(da, time_dim=time_dim),
            "RX5day": rx5day(da, time_dim=time_dim),
            "R95p": r95p(da, wet_threshold=wet_threshold, time_dim=time_dim),
            "PRCPTOT": prcptot(da, wet_threshold=wet_threshold, time_dim=time_dim),
        }
    )
