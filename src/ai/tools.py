"""Tool functions for the Gemini function-calling climate agent.

Each tool is a plain Python function with type hints and docstrings so that
Google Gemini can infer a schema from the source code.
"""

from pathlib import Path

import numpy as np
import xarray as xr

# Import from the project source tree
from src.diagnostics.extremes import r95p, rx1day, rx5day
from src.plotting.maps import (
    plot_bias_map,
    plot_map,
    plot_three_panel_climatology,
    save_figure,
)


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

# 20-year (1995-2014), Germany-only, monthly-mean gridded fields.
# Used by both the Regional Time-Series trend tool (spatial mean computed
# on the fly) and the Regional Climatology Map tool (kept as a 2D field).
DEMO_GERMANY_MONTHLY = {
    "cpc": DEMO_DIR / "cpc_germany_1995_2014_monthly.nc",
    "cmip6": DEMO_DIR / "cmip6_germany_1995_2014_monthly.nc",
}
DEMO_GERMANY_BBOX = [5.5, 15.5, 47.0, 55.5]


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


def regional_precip_trend(
    dataset: str,
    aggregation: str = "annual",
    output_dir: str | None = None,
) -> str:
    """Compute and plot a 20-year (1995-2014) precipitation trend for Germany.

    Uses the pre-downloaded, Germany-only, spatially-averaged daily
    precipitation time series (CPC reference and/or CMIP6 MPI-ESM1-2-HR
    simulation). This tool is limited to Germany and to the 1995-2014
    period, since that is the only multi-year data currently bundled with
    the demo.

    Parameters
    ----------
    dataset
        Which dataset(s) to plot: "cpc" (observed reference), "cmip6"
        (model simulation), or "both" (overlay both for comparison).
    aggregation
        Temporal aggregation of the trend: "annual" (annual mean,
        recommended for a clear trend line) or "monthly" (full monthly
        mean time series).
    output_dir
        Optional directory to save the figure. Defaults to the project's
        `figures/` folder.

    Returns
    -------
    str
        Confirmation message with the figure path and the computed linear
        trend slope(s) in mm/day per year.
    """
    import matplotlib.pyplot as plt

    dataset = dataset.lower().strip()
    aggregation = aggregation.lower().strip()
    if dataset not in ("cpc", "cmip6", "both"):
        raise ValueError(f"dataset must be 'cpc', 'cmip6', or 'both', got {dataset}")
    if aggregation not in ("annual", "monthly"):
        raise ValueError(f"aggregation must be 'annual' or 'monthly', got {aggregation}")

    keys = ["cpc", "cmip6"] if dataset == "both" else [dataset]

    series = {}
    for key in keys:
        path = DEMO_GERMANY_MONTHLY[key]
        ds = xr.open_dataset(path)
        var = _find_precip_variable(ds)
        da = ds[var]
        if "lat" in da.dims and "lon" in da.dims:
            da = da.mean(dim=["lat", "lon"])
        # Data is already monthly-mean; only resample further for annual aggregation
        if aggregation == "annual":
            da = da.resample(time="YS").mean()
        series[key] = da
        ds.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    trend_lines = []
    trend_slopes = {}
    labels = {"cpc": "CPC (observed)", "cmip6": "CMIP6 MPI-ESM1-2-HR (model)"}
    for key, da in series.items():
        x = da["time"].values
        y = da.values
        ax.plot(x, y, marker="o", markersize=3, label=labels[key])

        # Linear trend (mm/day per year)
        years = (da["time"].dt.year + da["time"].dt.dayofyear / 365.25).values
        slope, intercept = np.polyfit(years, y, 1)
        ax.plot(x, slope * years + intercept, linestyle="--", alpha=0.7,
                 label=f"{labels[key]} trend: {slope:+.4f} mm/day/yr")
        trend_lines.append(f"{labels[key]}: {slope:+.4f} mm/day per year")
        trend_slopes[key] = round(float(slope), 5)

    ax.set_title(f"Germany precipitation trend ({aggregation} mean, 1995-2014)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Precipitation (mm/day)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_dir is None:
        out_dir = FIGURES_DIR
    else:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"agent_regional_trend_{dataset}_{aggregation}_output.png"

    save_figure(
        fig,
        out_path,
        metadata={
            "variable": "precipitation",
            "diagnostic": f"Germany precipitation trend ({aggregation}, 1995-2014)",
            "dataset": dataset,
            "region": "Germany",
            "bbox": DEMO_GERMANY_BBOX,
            "period": "1995-2014",
            "units": "mm/day",
            "trend_mm_per_day_per_year": trend_slopes,
        },
    )

    trend_summary = "; ".join(trend_lines)
    return (
        f"Successfully computed the Germany {aggregation} precipitation trend "
        f"(1995-2014) for {dataset} and saved figure to {out_path}. "
        f"Linear trend(s): {trend_summary}."
    )


def germany_climatology_map(
    metric: str = "mean",
    dataset: str = "both",
    output_dir: str | None = None,
) -> str:
    """Plot a spatial precipitation climatology or bias map over Germany.

    Uses the 20-year (1995-2014) monthly gridded fields. When
    `dataset="both"`, a single 3-panel figure is produced: CPC mean,
    CMIP6 mean, and CMIP6-minus-CPC bias. With `dataset="cpc"` or
    `"cmip6"`, a single time-mean map is produced. `metric="bias"` always
    produces a single bias map.
    """
    metric = metric.lower().strip()
    dataset = dataset.lower().strip()
    if metric not in ("mean", "bias"):
        raise ValueError(f"metric must be 'mean' or 'bias', got {metric}")
    if dataset not in ("cpc", "cmip6", "both"):
        raise ValueError(f"dataset must be 'cpc', 'cmip6', or 'both', got {dataset}")

    if output_dir is None:
        out_dir = FIGURES_DIR
    else:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cpc = xr.open_dataset(DEMO_GERMANY_MONTHLY["cpc"])
    ds_cmip6 = xr.open_dataset(DEMO_GERMANY_MONTHLY["cmip6"])
    var_cpc = _find_precip_variable(ds_cpc)
    var_cmip6 = _find_precip_variable(ds_cmip6)
    clim_cpc = ds_cpc[var_cpc].mean(dim="time")
    clim_cmip6 = ds_cmip6[var_cmip6].mean(dim="time")
    clim_cmip6_regridded = clim_cmip6.interp(lat=clim_cpc["lat"], lon=clim_cpc["lon"])
    bias = clim_cmip6_regridded - clim_cpc
    bias.name = "precip_bias"

    out_path = out_dir / f"agent_germany_climatology_{dataset}_{metric}_output.png"

    if metric == "mean" and dataset == "both":
        fig = plot_three_panel_climatology(
            clim_cpc,
            clim_cmip6_regridded,
            bias,
            region="Germany",
            period="1995-2014",
            cbar_label="mm/day",
        )
        save_figure(
            fig,
            out_path,
            metadata={
                "variable": "precipitation",
                "diagnostic": "Germany mean precipitation (CPC, CMIP6, bias, 1995-2014)",
                "dataset": "cmip6 - cpc",
                "region": "Germany",
                "bbox": DEMO_GERMANY_BBOX,
                "period": "1995-2014",
                "units": "mm/day",
            },
        )
        ds_cpc.close()
        ds_cmip6.close()
        return f"Successfully computed the 3-panel Germany climatology figure and saved it to {out_path}"

    if metric == "mean":
        key = dataset
        clim = {"cpc": clim_cpc, "cmip6": clim_cmip6}[key]
        ds_cpc.close()
        ds_cmip6.close()
        fig = plot_map(
            clim,
            title=f"Germany mean precipitation ({key.upper()}, 1995-2014)",
            cbar_label="mm/day",
        )
        save_figure(
            fig,
            out_path,
            metadata={
                "variable": "precipitation",
                "diagnostic": f"Germany mean precipitation climatology ({key.upper()}, 1995-2014)",
                "dataset": key,
                "region": "Germany",
                "bbox": DEMO_GERMANY_BBOX,
                "period": "1995-2014",
                "units": "mm/day",
            },
        )
        return f"Successfully computed Germany mean precipitation climatology figure: {out_path}"

    # metric == "bias"
    ds_cpc.close()
    ds_cmip6.close()
    fig = plot_bias_map(
        bias,
        title="Germany mean precipitation bias: CMIP6 MPI-ESM1-2-HR minus CPC (1995-2014)",
        cbar_label="mm/day",
    )
    save_figure(
        fig,
        out_path,
        metadata={
            "variable": "precipitation",
            "diagnostic": "Germany mean precipitation bias (CMIP6 - CPC, 1995-2014)",
            "dataset": "cmip6 - cpc",
            "region": "Germany",
            "bbox": DEMO_GERMANY_BBOX,
            "period": "1995-2014",
            "units": "mm/day",
        },
    )
    return f"Successfully computed the Germany mean precipitation bias (CMIP6 - CPC) and saved figure to {out_path}"


def global_climatology_map(
    metric: str = "mean",
    dataset: str = "both",
    output_dir: str | None = None,
) -> str:
    """Plot a global precipitation climatology map or bias map.

    Uses the 2013 monthly global fields (CPC regridded to the CMIP6 grid and
    CMIP6 MPI-ESM1-2-HR). World maps are drawn in Robinson projection. When
    `dataset="both"`, a single 3-panel figure is produced: CPC mean,
    CMIP6 mean, and CMIP6-minus-CPC bias. With `dataset="cpc"` or
    `"cmip6"`, a single time-mean map is produced. `metric="bias"` always
    produces a single global bias map.
    """
    metric = metric.lower().strip()
    dataset = dataset.lower().strip()
    if metric not in ("mean", "bias"):
        raise ValueError(f"metric must be 'mean' or 'bias', got {metric}")
    if dataset not in ("cpc", "cmip6", "both"):
        raise ValueError(f"dataset must be 'cpc', 'cmip6', or 'both', got {dataset}")

    if output_dir is None:
        out_dir = FIGURES_DIR
    else:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cpc = xr.open_dataset(DEMO_MONTHLY["cpc"])
    ds_cmip6 = xr.open_dataset(DEMO_MONTHLY["cmip6"])
    var_cpc = _find_precip_variable(ds_cpc)
    var_cmip6 = _find_precip_variable(ds_cmip6)

    clim_cpc = ds_cpc[var_cpc].mean(dim="time")
    clim_cmip6 = ds_cmip6[var_cmip6].mean(dim="time")
    # The CPC global file is already regridded to the CMIP6 grid, so a direct
    # difference is safe. If they ever differ, fall back to interpolation.
    try:
        bias = clim_cmip6 - clim_cpc
    except Exception:
        clim_cmip6_regridded = clim_cmip6.interp(
            lat=clim_cpc["lat"], lon=clim_cpc["lon"]
        )
        bias = clim_cmip6_regridded - clim_cpc
    bias.name = "precip_bias"

    out_path = out_dir / f"agent_global_climatology_{dataset}_{metric}_output.png"

    if metric == "mean" and dataset == "both":
        fig = plot_three_panel_climatology(
            clim_cpc,
            clim_cmip6,
            bias,
            region="Global",
            period="2013",
            cbar_label="mm/day",
        )
        save_figure(
            fig,
            out_path,
            metadata={
                "variable": "precipitation",
                "diagnostic": "Global mean precipitation (CPC, CMIP6, bias, 2013)",
                "dataset": "cmip6 - cpc",
                "region": "Global",
                "period": "2013",
                "units": "mm/day",
            },
        )
        ds_cpc.close()
        ds_cmip6.close()
        return f"Successfully computed the 3-panel global climatology figure and saved it to {out_path}"

    if metric == "mean":
        key = dataset
        clim = {"cpc": clim_cpc, "cmip6": clim_cmip6}[key]
        ds_cpc.close()
        ds_cmip6.close()
        fig = plot_map(
            clim,
            title=f"Global mean precipitation ({key.upper()}, 2013)",
            cbar_label="mm/day",
        )
        save_figure(
            fig,
            out_path,
            metadata={
                "variable": "precipitation",
                "diagnostic": f"Global mean precipitation climatology ({key.upper()}, 2013)",
                "dataset": key,
                "region": "Global",
                "period": "2013",
                "units": "mm/day",
            },
        )
        return f"Successfully computed global mean precipitation climatology figure: {out_path}"

    # metric == "bias"
    ds_cpc.close()
    ds_cmip6.close()
    fig = plot_bias_map(
        bias,
        title="Global mean precipitation bias: CMIP6 MPI-ESM1-2-HR minus CPC (2013)",
        cbar_label="mm/day",
    )
    save_figure(
        fig,
        out_path,
        metadata={
            "variable": "precipitation",
            "diagnostic": "Global mean precipitation bias (CMIP6 - CPC, 2013)",
            "dataset": "cmip6 - cpc",
            "region": "Global",
            "period": "2013",
            "units": "mm/day",
        },
    )
    return f"Successfully computed the global mean precipitation bias (CMIP6 - CPC) and saved figure to {out_path}"


def compare_precip_at_point(
    lat: float,
    lon: float,
    dataset: str = "both",
    aggregation: str = "monthly",
    location_name: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Compare observed vs. simulated precipitation at a specific point (e.g. a city).

    Selects the nearest grid cell to the given latitude/longitude from the
    20-year (1995-2014) monthly gridded Germany fields and plots the
    resulting time series. Useful for city-level questions (e.g. "compare
    observed and modelled precipitation at Frankfurt"), as long as the point
    falls within Germany (the only region with a multi-year record bundled
    with the demo).

    Parameters
    ----------
    lat, lon
        Latitude and longitude (degrees) of the point of interest, e.g.
        Frankfurt am Main is approximately lat=50.11, lon=8.68.
    dataset
        Which dataset(s) to plot: "cpc" (observed), "cmip6" (model), or
        "both" (overlay both for comparison).
    aggregation
        "monthly" (native resolution) or "annual" (annual mean).
    location_name
        Optional human-readable name of the point (e.g. "Frankfurt") used in
        the plot title.
    output_dir
        Optional directory to save the figure. Defaults to the project's
        `figures/` folder.

    Returns
    -------
    str
        Confirmation message with the figure path, or an explanation if the
        point falls outside the available data extent.
    """
    import matplotlib.pyplot as plt

    dataset = dataset.lower().strip()
    aggregation = aggregation.lower().strip()
    if dataset not in ("cpc", "cmip6", "both"):
        raise ValueError(f"dataset must be 'cpc', 'cmip6', or 'both', got {dataset}")
    if aggregation not in ("annual", "monthly"):
        raise ValueError(f"aggregation must be 'annual' or 'monthly', got {aggregation}")

    min_lon, max_lon, min_lat, max_lat = DEMO_GERMANY_BBOX
    if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
        return (
            f"The point (lat={lat}, lon={lon}) falls outside the only multi-year "
            f"dataset extent available in this demo (Germany, bbox {DEMO_GERMANY_BBOX}). "
            "No figure was generated. Try a point within Germany."
        )

    keys = ["cpc", "cmip6"] if dataset == "both" else [dataset]
    labels = {"cpc": "CPC (observed)", "cmip6": "CMIP6 MPI-ESM1-2-HR (model)"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for key in keys:
        ds = xr.open_dataset(DEMO_GERMANY_MONTHLY[key])
        var = _find_precip_variable(ds)
        da = ds[var].sel(lat=lat, lon=lon, method="nearest")
        ds.close()

        if aggregation == "annual":
            da = da.resample(time="YS").mean()

        actual_lat = float(da["lat"].values)
        actual_lon = float(da["lon"].values)
        ax.plot(da["time"].values, da.values, marker="o", markersize=3, label=labels[key])

    place = location_name or f"({lat:.2f}, {lon:.2f})"
    ax.set_title(f"Precipitation at {place} — nearest grid cell ({aggregation} mean, 1995-2014)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Precipitation (mm/day)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_dir is None:
        out_dir = FIGURES_DIR
    else:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = (location_name or f"{lat:.2f}_{lon:.2f}").lower().replace(" ", "_")
    out_path = out_dir / f"agent_point_compare_{slug}_{aggregation}_output.png"

    save_figure(
        fig,
        out_path,
        metadata={
            "variable": "precipitation",
            "diagnostic": f"Precipitation at {place} ({aggregation}, 1995-2014)",
            "dataset": dataset,
            "region": location_name or "Germany (point)",
            "point": [lat, lon],
            "nearest_grid_point": [actual_lat, actual_lon],
            "period": "1995-2014",
            "units": "mm/day",
        },
    )

    return (
        f"Successfully compared precipitation at {place} (nearest grid point: "
        f"lat={actual_lat:.2f}, lon={actual_lon:.2f}) and saved figure to {out_path}"
    )


if __name__ == "__main__":
    # Quick sanity test
    print(
        compute_and_plot_extreme(
            dataset_path="data/cpc/cpc_precip_2013.nc",
            index_name="RX1day",
            region_bbox=[60, 100, 5, 35],
        )
    )
