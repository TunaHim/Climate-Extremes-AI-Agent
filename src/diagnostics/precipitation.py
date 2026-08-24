"""Precipitation diagnostic functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def hourly_to_daily_precip(da: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    """Aggregate hourly precipitation to daily totals.

    Works for both NumPy datetime64 and cftime time coordinates.
    """
    time_values = da[time_dim].values
    try:
        new_times = pd.to_datetime(time_values)
    except (TypeError, ValueError):
        # cftime objects are not convertible by pd.to_datetime directly;
        # fall back to ISO-string conversion.
        new_times = pd.to_datetime(
            [t.strftime("%Y-%m-%dT%H:%M:%S") for t in time_values]
        )
    da = da.assign_coords({time_dim: new_times})
    return da.resample({time_dim: "1D"}).sum(dim=time_dim, skipna=False)


def monthly_climatology(da: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    """Compute monthly climatological mean."""
    return da.groupby(f"{time_dim}.month").mean(dim=time_dim, skipna=True)


def seasonal_mean(da: xr.DataArray, season: str, time_dim: str = "time") -> xr.DataArray:
    """Compute seasonal mean for a given season (e.g. 'JJA')."""
    return da.sel({time_dim: da[time_dim].dt.season == season}).mean(dim=time_dim, skipna=True)


def compute_bias(sim: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Compute bias: simulation minus reference."""
    # Align grids
    sim_aligned, ref_aligned = xr.align(sim, ref, join="inner")
    bias = sim_aligned - ref_aligned
    bias.attrs["units"] = sim.attrs.get("units", "")
    bias.attrs["long_name"] = "Bias (simulation - reference)"
    return bias


def compute_relative_bias(sim: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Compute relative bias in percent."""
    sim_aligned, ref_aligned = xr.align(sim, ref, join="inner")
    rel_bias = 100.0 * (sim_aligned - ref_aligned) / ref_aligned.where(ref_aligned != 0)
    rel_bias.attrs["units"] = "%"
    rel_bias.attrs["long_name"] = "Relative bias (%)"
    return rel_bias
