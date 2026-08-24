"""Map plotting functions."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter


def _guess_lonlat(da: xr.DataArray):
    """Return longitude and latitude coordinate names."""
    candidates = {
        "lon": ("lon", "longitude", "x"),
        "lat": ("lat", "latitude", "y"),
    }
    lon_name = next((c for c in da.coords if c in candidates["lon"]), None)
    lat_name = next((c for c in da.coords if c in candidates["lat"]), None)
    return lon_name, lat_name


def _lon_to_180(da: xr.DataArray) -> xr.DataArray:
    """Wrap longitude from 0..360 to -180..180 and sort, keeping lat order."""
    lon_name, _ = _guess_lonlat(da)
    if lon_name is None:
        return da
    da = da.assign_coords({lon_name: ((da[lon_name] + 180) % 360) - 180})
    return da.sortby(lon_name)


def _is_global(da: xr.DataArray) -> bool:
    """Return True for near-global fields where a Robinson projection fits."""
    lon_name, lat_name = _guess_lonlat(da)
    if lon_name is None or lat_name is None:
        return False
    lon = da[lon_name].values
    lat = da[lat_name].values
    lon_range = float(np.nanmax(lon) - np.nanmin(lon))
    lat_range = float(np.nanmax(lat) - np.nanmin(lat))
    return lon_range > 330 and lat_range > 150


def _choose_projection(da: xr.DataArray) -> ccrs.Projection:
    """Use PlateCarree for all maps so longitude/latitude axes can be labelled."""
    return ccrs.PlateCarree()


def _prepare_geo_axes(
    fig: plt.Figure,
    projection: ccrs.Projection,
    nrows: int = 1,
    ncols: int = 1,
    index: int = 1,
    is_global: bool = False,
):
    """Add a Cartopy-aware subplot and (for global maps) set the global extent."""
    ax = fig.add_subplot(nrows, ncols, index, projection=projection)
    if is_global:
        ax.set_global()
    return ax


def _add_lat_lon_ticks(
    ax: plt.Axes, projection: ccrs.Projection, labelsize: int = 10
) -> None:
    """Add sensible longitude/latitude tick labels on PlateCarree axes."""
    if not isinstance(projection, ccrs.PlateCarree):
        return
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_range = x1 - x0
    y_range = y1 - y0
    # 45-degree spacing for global maps, 10/5 for regional
    dx = 5 if x_range <= 25 else (10 if x_range <= 120 else 45)
    dy = 5 if y_range <= 25 else (10 if y_range <= 60 else 30)
    xticks = np.arange(dx * np.floor(x0 / dx), dx * np.ceil(x1 / dx) + dx, dx)
    yticks = np.arange(dy * np.floor(y0 / dy), dy * np.ceil(y1 / dy) + dy, dy)
    ax.set_xticks(xticks[(xticks >= x0) & (xticks <= x1)], crs=ccrs.PlateCarree())
    ax.set_yticks(yticks[(yticks >= y0) & (yticks <= y1)], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis="both", labelsize=labelsize)


def plot_map(
    da: xr.DataArray,
    title: str = "",
    cmap: str = "YlGnBu",
    levels: int = 20,
    extend: str = "neither",
    cbar_label: str = "",
    coastline_resolution: str = "110m",
    figsize: tuple = (10, 6),
    projection: ccrs.Projection | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    labelsize: int = 10,
    titlesize: int | None = None,
    cbar_labelsize: int | None = None,
) -> plt.Figure:
    """Plot a global/regional map of a 2D DataArray.

    All maps use PlateCarree so longitude/latitude axes can be labelled.
    """
    da = _lon_to_180(da)
    if projection is None:
        projection = _choose_projection(da)
        is_global = _is_global(da)
    else:
        is_global = isinstance(projection, (ccrs.Robinson, ccrs.Mollweide))

    fig = plt.figure(figsize=figsize)
    ax = _prepare_geo_axes(fig, projection, is_global=is_global)

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
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )

    ax.coastlines(resolution=coastline_resolution)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)

    fontsize = labelsize
    title_font = titlesize if titlesize is not None else fontsize
    cbar_font = cbar_labelsize if cbar_labelsize is not None else fontsize

    ax.set_xlabel("Longitude", fontsize=fontsize)
    ax.set_ylabel("Latitude", fontsize=fontsize)

    _add_lat_lon_ticks(ax, projection, labelsize=fontsize)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    cbar.ax.tick_params(labelsize=cbar_font)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=cbar_font)

    ax.set_title(title, fontsize=title_font)
    plt.tight_layout()
    return fig


def plot_bias_map(
    da: xr.DataArray,
    title: str = "Bias",
    cmap: str = "RdBu",
    symmetric: bool = True,
    cbar_label: str = "",
    figsize: tuple = (10, 6),
    projection: ccrs.Projection | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """Plot a bias map with diverging color scale.

    All maps use PlateCarree so longitude/latitude axes can be labelled.
    """
    da = _lon_to_180(da)
    if projection is None:
        projection = _choose_projection(da)
        is_global = _is_global(da)
    else:
        is_global = isinstance(projection, (ccrs.Robinson, ccrs.Mollweide))

    if vmin is None or vmax is None:
        auto_vmax = float(np.nanmax(np.abs(da.values)))
        if symmetric:
            vmin = -auto_vmax
            vmax = auto_vmax
        else:
            vmin = float(np.nanmin(da.values))
            vmax = auto_vmax

    fig = plt.figure(figsize=figsize)
    ax = _prepare_geo_axes(fig, projection, is_global=is_global)

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

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    if cbar_label:
        cbar.set_label(cbar_label)

    _add_lat_lon_ticks(ax, projection)

    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_three_panel_climatology(
    da_cpc: xr.DataArray,
    da_cmip6: xr.DataArray,
    da_bias: xr.DataArray,
    region: str = "",
    period: str = "",
    cbar_label: str = "mm/day",
    figsize: tuple = (18, 5),
    vmin: float | None = None,
    vmax: float | None = None,
    bias_vmin: float | None = None,
    bias_vmax: float | None = None,
) -> plt.Figure:
    """Plot a 3-panel figure: observed, model, and model-minus-observed bias.

    The first two panels share a common color scale. All panels use PlateCarree
    so longitude/latitude axes can be labelled.
    """
    da_cpc = _lon_to_180(da_cpc)
    da_cmip6 = _lon_to_180(da_cmip6)
    da_bias = _lon_to_180(da_bias)

    projection = _choose_projection(da_cpc)
    is_global = _is_global(da_cpc)

    fig = plt.figure(figsize=figsize)

    data_panels = [da_cpc, da_cmip6]
    titles = [
        f"CPC (observed){f' — {region}' if region else ''}",
        f"CMIP6 MPI-ESM1-2-HR (model){f' — {region}' if region else ''}",
    ]
    cmaps = ["YlGnBu", "YlGnBu"]

    # Shared colour scale for the two mean panels
    if vmin is None or vmax is None:
        all_values = np.concatenate([np.ravel(da_cpc.values), np.ravel(da_cmip6.values)])
        vmin = float(np.nanmin(all_values)) if vmin is None else vmin
        vmax = float(np.nanmax(all_values)) if vmax is None else vmax

    for i, (da, t, cmap) in enumerate(zip(data_panels, titles, cmaps), start=1):
        ax = _prepare_geo_axes(fig, projection, nrows=1, ncols=3, index=i, is_global=is_global)
        lon_name, lat_name = _guess_lonlat(da)
        im = ax.contourf(
            da[lon_name].values,
            da[lat_name].values,
            da.values,
            levels=20,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(resolution="110m")
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.set_title(t)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        _add_lat_lon_ticks(ax, projection)

        if cbar_label:
            cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
            cbar.set_label(cbar_label)

    # Bias panel
    ax = _prepare_geo_axes(fig, projection, nrows=1, ncols=3, index=3, is_global=is_global)
    if bias_vmin is None or bias_vmax is None:
        auto_vmax = float(np.nanmax(np.abs(da_bias.values)))
        bias_vmin = -auto_vmax if bias_vmin is None else bias_vmin
        bias_vmax = auto_vmax if bias_vmax is None else bias_vmax
    lon_name, lat_name = _guess_lonlat(da_bias)
    im = ax.contourf(
        da_bias[lon_name].values,
        da_bias[lat_name].values,
        da_bias.values,
        levels=20,
        cmap="RdBu",
        vmin=bias_vmin,
        vmax=bias_vmax,
        transform=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="110m")
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    bias_title = f"Bias: CMIP6 minus CPC{f' — {region}' if region else ''}"
    ax.set_title(bias_title)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    _add_lat_lon_ticks(ax, projection)

    if cbar_label:
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
        cbar.set_label(cbar_label)

    period_text = f" ({period})" if period else ""
    fig.suptitle(f"Mean precipitation{period_text}{f' — {region}' if region else ''}", y=1.02)
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
