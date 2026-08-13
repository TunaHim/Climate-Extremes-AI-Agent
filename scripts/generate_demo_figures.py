#!/usr/bin/env python3
"""Generate demo figures shipped with the repository for Streamlit Cloud."""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.ai.tools import compute_and_plot_extreme
from src.plotting.maps import plot_bias_map, plot_map, save_figure

DEMO_DIR = BASE_DIR / "data" / "demo"
FIGURES_DIR = BASE_DIR / "assets" / "demo_figures"


def annual_mean(path: Path, var: str) -> xr.DataArray:
    """Compute annual mean from a monthly dataset."""
    with xr.open_dataset(path) as ds:
        return ds[var].mean(dim="time", skipna=True)


def generate_mean_bias_figures() -> None:
    """Create CMIP6 mean, CPC mean, and model-minus-reference bias maps."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sim_mean = annual_mean(DEMO_DIR / "cmip6_global_monthly_2013.nc", "pr")
    ref_mean = annual_mean(DEMO_DIR / "cpc_global_monthly_2013_regridded.nc", "precip")

    # Ensure common grid before differencing
    if "latitude" in ref_mean.coords and "lat" not in ref_mean.coords:
        ref_mean = ref_mean.rename({"latitude": "lat"})
    if "longitude" in ref_mean.coords and "lon" not in ref_mean.coords:
        ref_mean = ref_mean.rename({"longitude": "lon"})
    sim_mean = sim_mean.interp_like(ref_mean, method="linear")
    bias = sim_mean - ref_mean

    fig = plot_map(
        sim_mean,
        title="MPI-ESM1-2-HR (CMIP6) mean daily precipitation 2013",
        cmap="YlGnBu",
        cbar_label="mm/day",
    )
    save_figure(
        fig,
        FIGURES_DIR / "demo_cmip6_mean_precip_2013.png",
        metadata={
            "diagnostic": "mean daily precipitation",
            "variable": "pr",
            "model": "MPI-ESM1-2-HR (CMIP6)",
            "period": "2013",
            "units": "mm/day",
        },
    )

    fig = plot_map(
        ref_mean,
        title="CPC (regridded) mean daily precipitation 2013",
        cmap="YlGnBu",
        cbar_label="mm/day",
    )
    save_figure(
        fig,
        FIGURES_DIR / "demo_cpc_mean_precip_2013.png",
        metadata={
            "diagnostic": "mean daily precipitation",
            "variable": "precip",
            "model": "CPC (regridded to CMIP6 grid)",
            "period": "2013",
            "units": "mm/day",
        },
    )

    fig = plot_bias_map(
        bias,
        title="Bias: CMIP6 minus CPC (mm/day)",
        cbar_label="mm/day",
    )
    save_figure(
        fig,
        FIGURES_DIR / "demo_bias_mean_precip_2013.png",
        metadata={
            "diagnostic": "mean daily precipitation bias",
            "variable": "pr",
            "models": "MPI-ESM1-2-HR (CMIP6) vs CPC",
            "period": "2013",
            "units": "mm/day",
        },
    )


def generate_extreme_figures() -> None:
    """Create regional extreme-index maps for the agent demo."""
    regions = {
        "south_asia": ("data/demo/cpc_south_asia_2013.nc", [60, 100, 5, 35]),
        "europe": ("data/demo/cpc_europe_2013.nc", [-15, 45, 35, 75]),
    }
    for region, (dataset, bbox) in regions.items():
        for index in ["RX1day", "RX5day", "R95p"]:
            out_name = f"demo_{region}_{index.lower()}_output.png"
            compute_and_plot_extreme(
                dataset_path=dataset,
                index_name=index,
                region_bbox=bbox,
                output_dir=str(FIGURES_DIR),
            )
            # Rename the generic agent output to a region-specific name
            src_png = FIGURES_DIR / f"agent_{index}_output.png"
            dst_png = FIGURES_DIR / out_name
            src_json = FIGURES_DIR / f"agent_{index}_output.json"
            dst_json = FIGURES_DIR / out_name.replace(".png", ".json")
            if src_png.exists():
                src_png.replace(dst_png)
            if src_json.exists():
                src_json.replace(dst_json)
            print(f"Saved {dst_png}")


def main():
    print("Generating demo figures in", FIGURES_DIR)
    print("\nMean and bias maps:")
    generate_mean_bias_figures()
    print("\nRegional extreme indices:")
    generate_extreme_figures()
    print("\nDone.")


if __name__ == "__main__":
    main()
