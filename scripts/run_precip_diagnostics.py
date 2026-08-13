#!/usr/bin/env python3
"""
Run the full precipitation diagnostics pipeline.

Assumes data have been downloaded to data/destine/ and data/era5/.
"""

import sys
from pathlib import Path

import xarray as xr

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diagnostics.extremes import compute_extreme_indices
from diagnostics.precipitation import compute_bias, compute_relative_bias, hourly_to_daily_precip
from plotting.maps import plot_bias_map, plot_map, save_figure
from utils.io import load_cmip6_precip, load_cpc_precip, load_destine_precip, load_era5_precip
from utils.regrid import regrid_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def find_first_file(directory: Path, pattern: str) -> Path | None:
    """Return first file matching a glob pattern."""
    matches = list(directory.rglob(pattern))
    return matches[0] if matches else None


def load_simulation_data():
    """Load the first available simulation dataset (DestinE or CMIP6)."""
    destine_file = find_first_file(DATA_DIR / "destine", "*.grib*")
    cmip6_file = find_first_file(DATA_DIR / "cmip6", "*.nc")

    if destine_file:
        print(f"Loading DestinE data from {destine_file}")
        da = load_destine_precip(destine_file)
        model_name = "IFS-FESOM (DestinE)"
        return hourly_to_daily_precip(da), model_name

    if cmip6_file:
        print(f"Loading CMIP6 data from {cmip6_file}")
        da = load_cmip6_precip(cmip6_file)
        model_name = "MPI-ESM1-2-HR (CMIP6)"
        return da, model_name

    raise FileNotFoundError(
        "No simulation data found. Run download_destine_precip.py or download_cmip6_precip.py first."
    )


def load_reference_data():
    """Load the first available reference dataset (ERA5 or CPC)."""
    era5_file = find_first_file(DATA_DIR / "era5", "*.nc")
    if era5_file:
        print(f"Loading ERA5 data from {era5_file}")
        da = load_era5_precip(era5_file)
        return hourly_to_daily_precip(da), "ERA5"

    cpc_file = find_first_file(DATA_DIR / "cpc", "*.nc")
    if cpc_file:
        print(f"Loading CPC data from {cpc_file}")
        return load_cpc_precip(cpc_file), "CPC"

    raise FileNotFoundError(
        "No reference data found. Run download_era5_precip.py or download_cpc_precip.py first."
    )


def subset_month(da: xr.DataArray, year: int, month: int) -> xr.DataArray:
    """Select a specific year and month from a daily time series."""
    return da.sel(time=(da["time"].dt.year == year) & (da["time"].dt.month == month))


def main():
    sim_daily_raw, model_name = load_simulation_data()
    ref_daily_raw, ref_name = load_reference_data()

    print("Subsetting to target month: 2013-07")
    sim_daily = subset_month(sim_daily_raw, 2013, 7)
    ref_daily = subset_month(ref_daily_raw, 2013, 7)

    print("Regridding to common 0.5 degree grid")
    sim_regrid = regrid_dataset(sim_daily.to_dataset(name="tp"), resolution=0.5)
    ref_regrid = regrid_dataset(ref_daily.to_dataset(name="tp"), resolution=0.5)

    sim_tp = sim_regrid["tp"]
    ref_tp = ref_regrid["tp"]

    # --- Mean precipitation map ---
    sim_mean = sim_tp.mean(dim="time", skipna=True)
    ref_mean = ref_tp.mean(dim="time", skipna=True)
    bias_mean = compute_bias(sim_mean, ref_mean)

    fig = plot_map(
        sim_mean,
        title=f"{model_name} mean daily precipitation (mm/day)",
        cmap="YlGnBu",
        cbar_label="mm/day",
    )
    save_figure(fig, FIGURES_DIR / "sim_mean_precip.png", metadata={
        "variable": "total precipitation",
        "units": "mm/day",
        "model": model_name,
        "period": "2013-07",
        "diagnostic": "mean daily precipitation",
    })

    fig = plot_map(
        ref_mean,
        title=f"{ref_name} mean daily precipitation (mm/day)",
        cmap="YlGnBu",
        cbar_label="mm/day",
    )
    save_figure(fig, FIGURES_DIR / "ref_mean_precip.png", metadata={
        "variable": "total precipitation",
        "units": "mm/day",
        "model": ref_name,
        "period": "2013-07",
        "diagnostic": "mean daily precipitation",
    })

    fig = plot_bias_map(
        bias_mean,
        title=f"Bias: {model_name} minus {ref_name} (mm/day)",
        cbar_label="mm/day",
    )
    save_figure(fig, FIGURES_DIR / "bias_mean_precip.png", metadata={
        "variable": "total precipitation bias",
        "units": "mm/day",
        "models": f"{model_name} vs {ref_name}",
        "period": "2013-07",
        "diagnostic": "mean daily precipitation bias",
    })

    # --- Extreme indices ---
    print("Computing extreme indices")
    sim_extremes = compute_extreme_indices(sim_tp, wet_threshold=1.0)
    ref_extremes = compute_extreme_indices(ref_tp, wet_threshold=1.0)

    for index in sim_extremes.data_vars:
        bias_extreme = compute_bias(sim_extremes[index], ref_extremes[index])
        fig = plot_bias_map(
            bias_extreme,
            title=f"Bias in {index}: {model_name} minus {ref_name}",
            cbar_label="mm",
        )
        save_figure(fig, FIGURES_DIR / f"bias_{index.lower()}.png", metadata={
            "variable": "precipitation",
            "index": index,
            "units": "mm",
            "models": f"{model_name} vs {ref_name}",
            "period": "2013-07",
            "diagnostic": f"{index} bias",
        })

    print(f"Figures saved to {FIGURES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
