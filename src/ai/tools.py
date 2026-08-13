"""Tool functions for the Gemini function-calling climate agent.

Each tool is a plain Python function with type hints and docstrings so that
Google Gemini can infer a schema from the source code.
"""

from pathlib import Path

import xarray as xr

# Import from the project source tree
from src.diagnostics.extremes import r95p, rx1day, rx5day
from src.plotting.maps import plot_map, save_figure


BASE_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = BASE_DIR / "figures"
DEMO_DIR = BASE_DIR / "data" / "demo"
DEMO_FIGURES_DIR = BASE_DIR / "assets" / "demo_figures"

# Lightweight datasets shipped with the repository for Streamlit Cloud
DEMO_DATASET = DEMO_DIR / "cpc_south_asia_2013.nc"
DEMO_REGION_BBOX = [60.0, 100.0, 5.0, 35.0]

DEMO_REGIONS = {
    "south_asia": {
        "path": DEMO_DIR / "cpc_south_asia_2013.nc",
        "bbox": [60.0, 100.0, 5.0, 35.0],
        "description": "South Asia",
    },
    "europe": {
        "path": DEMO_DIR / "cpc_europe_2013.nc",
        "bbox": [-15.0, 45.0, 35.0, 75.0],
        "description": "Europe",
    },
}

DEMO_MONTHLY = {
    "cmip6": DEMO_DIR / "cmip6_global_monthly_2013.nc",
    "cpc": DEMO_DIR / "cpc_global_monthly_2013_regridded.nc",
}


def _find_precip_variable(ds: xr.Dataset) -> str:
    """Return the name of the precipitation variable in a dataset."""
    candidates = ["precip", "pr", "tp", "rain", "precipitation"]
    for var in ds.data_vars:
        if var.lower() in candidates:
            return var
    # Fallback: take the first variable that has lat/lon/time dims
    for var in ds.data_vars:
        dims = {d.lower() for d in ds[var].dims}
        if {"time", "lat", "lon"} <= dims or {"time", "latitude", "longitude"} <= dims:
            return var
    raise ValueError("Could not identify a precipitation variable in the dataset.")


def _standardise_coords(da: xr.DataArray) -> xr.DataArray:
    """Rename latitude/longitude coordinates to lat/lon for consistency."""
    rename = {}
    for old, new in [("latitude", "lat"), ("longitude", "lon")]:
        if old in da.coords and new not in da.coords:
            rename[old] = new
    if rename:
        da = da.rename(rename)
    return da


def _slice_region(da: xr.DataArray, region_bbox: list[float]) -> xr.DataArray:
    """Slice a DataArray to a geographic bounding box.

    Parameters
    ----------
    region_bbox
        [min_lon, max_lon, min_lat, max_lat]. Longitude must be in the same
        0-360 or -180-180 range as the dataset coordinates.
    """
    # Ensure region_bbox is a plain list of floats (not a protobuf composite)
    region_bbox = [float(v) for v in region_bbox]
    min_lon, max_lon, min_lat, max_lat = region_bbox
    da = _standardise_coords(da)

    # Handle descending latitude (common in gridded observations like CPC)
    lat_values = da["lat"].values
    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(max_lat, min_lat)
    else:
        lat_slice = slice(min_lat, max_lat)

    lon_slice = slice(min_lon, max_lon)

    return da.sel(lat=lat_slice, lon=lon_slice)


def calculate_rx1day(da: xr.DataArray) -> xr.DataArray:
    """Calculate the RX1day index: maximum 1-day precipitation amount."""
    return rx1day(da)


def calculate_rx5day(da: xr.DataArray) -> xr.DataArray:
    """Calculate the RX5day index: maximum 5-day consecutive precipitation amount."""
    return rx5day(da)


def calculate_r95p(da: xr.DataArray) -> xr.DataArray:
    """Calculate the R95p index: total precipitation from wet days above the 95th percentile."""
    return r95p(da)


def compute_and_plot_extreme(
    dataset_path: str,
    index_name: str,
    region_bbox: list[float],
    output_dir: str | None = None,
) -> str:
    """Load a NetCDF dataset, compute a precipitation extreme index over a region, and save a map.

    Parameters
    ----------
    dataset_path
        Absolute or relative path to the NetCDF file containing daily precipitation.
    index_name
        One of "RX1day", "RX5day", or "R95p".
    region_bbox
        Bounding box as [min_lon, max_lon, min_lat, max_lat].
    output_dir
        Optional directory to save the figure. Defaults to the project's `figures/` folder.

    Returns
    -------
    str
        Confirmation message with the path to the saved figure.
    """
    canonical = {"RX1DAY": "RX1day", "RX5DAY": "RX5day", "R95P": "R95p"}
    lookup = index_name.upper()
    if lookup not in canonical:
        valid = ", ".join(canonical.values())
        raise ValueError(f"index_name must be one of {valid}, got {index_name}")
    index_name = canonical[lookup]

    # Ensure protobuf/JSON inputs are plain Python types
    region_bbox = [float(v) for v in region_bbox]

    path = Path(dataset_path)
    if not path.is_absolute():
        path = BASE_DIR / path

    ds = xr.open_dataset(path)
    var = _find_precip_variable(ds)
    da = ds[var]

    # Subset to region
    da_region = _slice_region(da, region_bbox)

    # Compute index
    index_dispatch = {
        "RX1day": calculate_rx1day,
        "RX5day": calculate_rx5day,
        "R95p": calculate_r95p,
    }
    result = index_dispatch[index_name](da_region)

    # If the result still has a time/year dimension, select the first slice for plotting
    if "time" in result.dims:
        result = result.isel(time=0)
    if "year" in result.dims:
        result = result.isel(year=0)

    # Plot and save
    if output_dir is None:
        out_dir = FIGURES_DIR
    else:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"agent_{index_name}_output.png"

    fig = plot_map(
        result,
        title=f"{index_name} over region {region_bbox}",
        cbar_label="mm",
    )
    save_figure(
        fig,
        out_path,
        metadata={
            "variable": "precipitation",
            "index": index_name,
            "units": "mm",
            "dataset": str(path),
            "region_bbox": region_bbox,
            "diagnostic": f"{index_name} over selected region",
        },
    )

    return f"Successfully calculated {index_name} and saved figure to {out_path}"


if __name__ == "__main__":
    # Quick sanity test
    print(
        compute_and_plot_extreme(
            dataset_path="data/cpc/cpc_precip_2013.nc",
            index_name="RX1day",
            region_bbox=[60, 100, 5, 35],
        )
    )
