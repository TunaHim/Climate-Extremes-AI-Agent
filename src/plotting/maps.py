"""Map plotting functions."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _guess_lonlat(da: xr.DataArray):
    """Return longitude and latitude coordinate names."""
    candidates = {
        "lon": ("lon", "longitude", "x"),
        "lat": ("lat", "latitude", "y"),
    }
    lon_name = next((c for c in da.coords if c in candidates["lon"]), None)
    lat_name = next((c for c in da.coords if c in candidates["lat"]), None)
    return lon_name, lat_name


def plot_map(
    da: xr.DataArray,
    title: str = "",
    cmap: str = "YlGnBu",
    levels: int = 20,
    extend: str = "neither",
    cbar_label: str = "",
    coastline_resolution: str = "110m",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot a global/regional map of a 2D DataArray."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    lon_name, lat_name = _guess_lonlat(da)
    lon = da[lon_name].values
    lat = da[lat_name].values

    im = ax.contourf(
        lon,
        lat,
        da.values,
        levels=levels,
        cmap=cmap,
        extend=extend,
        transform=ccrs.PlateCarree(),
    )

    ax.coastlines(resolution=coastline_resolution)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    if cbar_label:
        cbar.set_label(cbar_label)

    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_bias_map(
    da: xr.DataArray,
    title: str = "Bias",
    cmap: str = "RdBu",
    symmetric: bool = True,
    cbar_label: str = "",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot a bias map with diverging color scale."""
    vmax = float(np.nanmax(np.abs(da.values)))
    if symmetric:
        vmin = -vmax
    else:
        vmin = float(np.nanmin(da.values))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    lon_name, lat_name = _guess_lonlat(da)
    lon = da[lon_name].values
    lat = da[lat_name].values

    im = ax.contourf(
        lon,
        lat,
        da.values,
        levels=20,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )

    ax.coastlines(resolution="110m")
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    if cbar_label:
        cbar.set_label(cbar_label)

    ax.set_title(title)
    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150, metadata: dict = None) -> None:
    """Save a figure and optionally write a JSON sidecar with metadata."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if metadata:
        sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps(metadata, indent=2))
