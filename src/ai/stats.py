"""Statistical diagnostic tools for the climate ReAct agent.

These functions complement the existing plotting tools by returning quantitative
measures (regression p-values, correlations, bias metrics, distribution fits)
that the agent can use in its summaries.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import xarray as xr

from src.ai.tools import BASE_DIR, DEMO_GERMANY_BBOX, FIGURES_DIR, _find_precip_variable, _slice_region


SUPPORTED_DISTRIBUTIONS = {
    "gamma": st.gamma,
    "weibull_min": st.weibull_min,
    "gumbel_r": st.gumbel_r,
    "norm": st.norm,
    "lognorm": st.lognorm,
    "expon": st.expon,
}


def _open_dataset(path: str) -> xr.Dataset:
    """Open a dataset relative to the project root or as an absolute path."""
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    return xr.open_dataset(p)


def _resolve_variable(ds: xr.Dataset, variable: str | None) -> str:
    """Return the requested or auto-detected precipitation variable name."""
    if variable is None:
        return _find_precip_variable(ds)
    return variable


def _resolve_dataarray(
    dataset_path: str,
    variable: str | None = None,
    region_bbox: list[float] | None = None,
    point: list[float] | None = None,
    aggregation: str | None = None,
    time_slice: str | None = None,
) -> tuple[xr.DataArray, str, str]:
    """Load, subset and possibly aggregate a variable from a dataset.

    Returns the DataArray, a human-readable description, and the description
    of the spatial / temporal slice used.
    """
    ds = _open_dataset(dataset_path)
    try:
        var = _resolve_variable(ds, variable)
        da = ds[var]
    finally:
        ds.close()

    # Standardise coords
    da = _standardise_coords(da)

    if time_slice:
        da = da.sel(time=slice(*time_slice.split(",")))  # type: ignore[arg-type]

    if point is not None:
        lat, lon = float(point[0]), float(point[1])
        da = da.sel(lat=lat, lon=lon, method="nearest")
        desc = f"point ({lat:.2f}, {lon:.2f})"
    elif region_bbox is not None:
        da = _slice_region(da, region_bbox)
        desc = f"region bbox {region_bbox}"
    else:
        desc = "full spatial domain"

    if aggregation:
        agg = aggregation.lower().strip()
        if agg == "annual":
            da = da.resample(time="YS").mean()
        elif agg == "monthly":
            da = da.resample(time="MS").mean()
        elif agg == "daily":
            pass
        elif agg == "mean":
            da = da.mean(dim="time")
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

    return da, var, desc


def _standardise_coords(da: xr.DataArray) -> xr.DataArray:
    """Rename latitude/longitude coordinates to lat/lon for consistency."""
    rename = {}
    for old, new in [("latitude", "lat"), ("longitude", "lon")]:
        if old in da.coords and new not in da.coords:
            rename[old] = new
    if rename:
        da = da.rename(rename)
    return da


def _save_or_return(
    fig: plt.Figure,
    slug: str,
    output_dir: str | None,
    metadata: dict,
) -> Path:
    """Save a figure and return the path."""
    from src.plotting.maps import save_figure

    if output_dir is None:
        out_dir = FIGURES_DIR
    else:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"agent_{slug}_output.png"
    save_figure(fig, out_path, metadata=metadata)
    return out_path


def _time_to_numeric(da: xr.DataArray) -> np.ndarray:
    """Convert a time coordinate to a floating-point year value."""
    year = da["time"].dt.year.values.astype(float)
    doy = da["time"].dt.dayofyear.values.astype(float)
    days_in_year = 365.25  # approximation sufficient for trend significance
    return year + doy / days_in_year


def linear_regression_analysis(
    dataset_path: str,
    variable: str | None = None,
    region_bbox: list[float] | None = None,
    point: list[float] | None = None,
    aggregation: str = "annual",
    output_dir: str | None = None,
) -> str:
    """Fit a linear trend to a precipitation time series and report statistics.

    Computes the slope, intercept, R², p-value and 95% confidence interval.
    The series can be a spatial mean over a region, or a single grid cell.

    Parameters
    ----------
    dataset_path
        Path to the NetCDF file.
    variable
        Variable name. If None, auto-detects the precipitation variable.
    region_bbox
        Bounding box [min_lon, max_lon, min_lat, max_lat] for a spatial mean.
    point
        [lat, lon] to select a single grid cell. Overrides region_bbox if both
        are provided.
    aggregation
        Temporal aggregation of the input before fitting: "annual", "monthly",
        "daily", or "mean" (collapses time entirely).
    output_dir
        Optional directory to save a time-series + trend figure.

    Returns
    -------
    str
        Trend statistics and, if requested, the saved figure path.
    """
    da, var, desc = _resolve_dataarray(
        dataset_path,
        variable,
        region_bbox,
        point,
        aggregation,
    )

    if aggregation is not None and aggregation.lower() == "mean":
        raise ValueError("aggregation='mean' collapses time; cannot fit a trend.")

    if "lat" in da.dims and "lon" in da.dims:
        da = da.mean(dim=["lat", "lon"], skipna=True)

    # Drop NaNs and align
    da = da.dropna("time")
    y = da.values.astype(float)
    x = _time_to_numeric(da)

    if len(y) < 3:
        return (
            f"Cannot compute linear regression for {desc}: only {len(y)} "
            "valid time steps available (need at least 3)."
        )

    result = st.linregress(x, y)
    slope = float(result.slope)
    intercept = float(result.intercept)
    rvalue = float(result.rvalue)
    pvalue = float(result.pvalue)
    stderr = float(result.stderr)

    # 95% CI for the slope
    df = max(len(y) - 2, 1)
    t = st.t.ppf(0.975, df)
    ci_lower = slope - t * stderr
    ci_upper = slope + t * stderr

    years = x
    trend = slope * years + intercept

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(da["time"].values, y, marker="o", markersize=3, label=var)
    ax.plot(da["time"].values, trend, "--", label=f"trend: {slope:+.4f} /yr")
    ax.set_title(f"Linear trend ({var}, {desc})")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{var}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = _save_or_return(
        fig,
        f"linear_trend_{var}_{desc.replace(' ', '_').replace(',', '')}",
        output_dir,
        metadata={
            "variable": var,
            "diagnostic": f"Linear trend for {var} over {desc}",
            "dataset": str(dataset_path),
            "slope_per_year": round(slope, 6),
            "r2": round(rvalue ** 2, 4),
            "p_value": round(pvalue, 6),
            "ci_95": [round(ci_lower, 6), round(ci_upper, 6)],
        },
    )

    return (
        f"Linear regression for {var} at {desc}: slope={slope:+.6f} per year, "
        f"R²={rvalue**2:.4f}, p-value={pvalue:.2e}, "
        f"95% CI=[{ci_lower:.6f}, {ci_upper:.6f}], "
        f"n={len(y)}. Figure saved to {out_path}."
    )


def _regrid_to_match(da: xr.DataArray, target_da: xr.DataArray) -> xr.DataArray:
    """Regrid a DataArray onto the lat/lon grid of a target DataArray."""
    if "lat" in da.dims and "lon" in da.dims:
        return da.interp(lat=target_da["lat"], lon=target_da["lon"])
    return da


def _flatten_spatial(
    da: xr.DataArray,
    weight_by_lat: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Flatten a spatial or spatio-temporal DataArray and return optional latitude weights."""
    dims = [d for d in da.dims if d in ("lat", "latitude", "lon", "longitude")]
    if not dims:
        return da.values.astype(float).ravel(), None

    stacked = da.stack(z=dims)
    vals = stacked.values.astype(float).ravel()

    weights = None
    if weight_by_lat and ("lat" in stacked.coords or "latitude" in stacked.coords):
        lat_name = "lat" if "lat" in stacked.coords else "latitude"
        lat = stacked[lat_name].values.ravel()
        weights = np.sqrt(np.abs(np.cos(np.radians(lat))))
        weights[np.isnan(weights)] = 0.0

    return vals, weights


def spatial_pattern_correlation(
    dataset_a: str,
    dataset_b: str,
    variable: str | None = None,
    region_bbox: list[float] | None = None,
    weight_by_lat: bool = True,
    time_index: int | None = None,
) -> str:
    """Compute Pearson and area-weighted pattern correlation between two fields.

    Parameters
    ----------
    dataset_a
        Path to the reference NetCDF.
    dataset_b
        Path to the second NetCDF (e.g. model output). Regridded onto A's grid.
    variable
        Variable name. If None, auto-detected from both files.
    region_bbox
        Optional bounding box [min_lon, max_lon, min_lat, max_lat].
    weight_by_lat
        If True, pattern correlation uses sqrt(cos(lat)) area weighting.
    time_index
        For 3D data, the index of the time slice to use. If None, the time mean
        is taken first.

    Returns
    -------
    str
        Pearson r, p-value, pattern correlation and sample size.
    """
    da_a, var_a, desc = _resolve_dataarray(dataset_a, variable, region_bbox)
    ds_b = _open_dataset(dataset_b)
    try:
        var_b = _resolve_variable(ds_b, variable)
        da_b = _standardise_coords(ds_b[var_b])
    finally:
        ds_b.close()

    if region_bbox:
        da_b = _slice_region(da_b, region_bbox)

    # Align on time if possible
    if "time" in da_a.dims and "time" in da_b.dims:
        if time_index is not None:
            da_a = da_a.isel(time=time_index)
            da_b = da_b.isel(time=time_index)
        else:
            # Use the mean over time to focus on spatial patterns
            da_a = da_a.mean(dim="time", skipna=True)
            da_b = da_b.mean(dim="time", skipna=True)

    # Regrid B onto A
    if "lat" in da_a.dims and "lon" in da_a.dims:
        da_b = _regrid_to_match(da_b, da_a)

    # Align coordinates
    da_b = da_b.reindex_like(da_a, method="nearest")

    vals_a, _ = _flatten_spatial(da_a, weight_by_lat=False)
    vals_b, _ = _flatten_spatial(da_b, weight_by_lat=False)

    mask = np.isfinite(vals_a) & np.isfinite(vals_b)
    if mask.sum() < 3:
        return f"Cannot compute correlation for {desc}: fewer than 3 overlapping grid cells."

    a = vals_a[mask]
    b = vals_b[mask]

    pearson_r, pearson_p = st.pearsonr(a, b)

    # Area-weighted pattern correlation
    _, weights = _flatten_spatial(da_a, weight_by_lat=weight_by_lat)
    if weights is not None and weight_by_lat:
        w = weights[mask]
        w /= w.sum()
        wa = a - np.average(a, weights=w)
        wb = b - np.average(b, weights=w)
        numerator = np.sum(w * wa * wb)
        denom = np.sqrt(np.sum(w * wa ** 2)) * np.sqrt(np.sum(w * wb ** 2))
        pattern_corr = float(numerator / denom) if denom > 0 else np.nan
    else:
        pattern_corr = float(pearson_r)

    return (
        f"Spatial comparison for {var_a} over {desc}: "
        f"Pearson r={pearson_r:.4f} (p={pearson_p:.2e}), "
        f"pattern correlation={pattern_corr:.4f} (area-weighted={weight_by_lat}), "
        f"n_grid={int(mask.sum())}."
    )


def bias_metrics(
    dataset_a: str,
    dataset_b: str,
    variable: str | None = None,
    region_bbox: list[float] | None = None,
    output_dir: str | None = None,
) -> str:
    """Compute RMSE, MAE, mean error and other bias metrics between two fields.

    Parameters
    ----------
    dataset_a
        Path to the reference NetCDF.
    dataset_b
        Path to the second NetCDF (e.g. model output).
    variable
        Variable name. If None, auto-detected.
    region_bbox
        Optional bounding box [min_lon, max_lon, min_lat, max_lat].
    output_dir
        Optional directory to save a difference map figure.

    Returns
    -------
    str
        RMSE, MAE, ME, bias, standard deviation of error and figure path.
    """
    da_a, var_a, desc = _resolve_dataarray(dataset_a, variable, region_bbox)
    ds_b = _open_dataset(dataset_b)
    try:
        var_b = _resolve_variable(ds_b, variable)
        da_b = _standardise_coords(ds_b[var_b])
    finally:
        ds_b.close()

    if region_bbox:
        da_b = _slice_region(da_b, region_bbox)

    if "time" in da_a.dims and "time" in da_b.dims:
        # Compare mean spatial patterns
        da_a = da_a.mean(dim="time", skipna=True)
        da_b = da_b.mean(dim="time", skipna=True)

    if "lat" in da_a.dims and "lon" in da_a.dims:
        da_b = _regrid_to_match(da_b, da_a)
    da_b = da_b.reindex_like(da_a, method="nearest")

    vals_a, _ = _flatten_spatial(da_a, weight_by_lat=False)
    vals_b, _ = _flatten_spatial(da_b, weight_by_lat=False)

    mask = np.isfinite(vals_a) & np.isfinite(vals_b)
    if mask.sum() < 1:
        return f"Cannot compute bias metrics for {desc}: no overlapping valid grid cells."

    a = vals_a[mask]
    b = vals_b[mask]
    diff = b - a

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    me = float(np.mean(diff))
    std_err = float(np.std(diff, ddof=1))
    max_abs_err = float(np.max(np.abs(diff)))
    pearson_r, _ = st.pearsonr(a, b)

    # Optional difference map
    out_path = None
    if "lat" in da_a.dims and "lon" in da_a.dims:
        diff_da = da_b - da_a
        diff_da.name = f"{var_b}_minus_{var_a}_bias"
        fig, ax = plt.subplots(figsize=(9, 5))
        diff_da.plot(ax=ax, cmap="RdBu_r", center=0)
        ax.set_title(f"Bias: {var_b} - {var_a}")
        plt.tight_layout()
        out_path = _save_or_return(
            fig,
            f"bias_{var_b}_minus_{var_a}",
            output_dir,
            metadata={
                "variable": var_a,
                "diagnostic": f"Bias metrics and map ({var_b} - {var_a}) over {desc}",
                "rmse": round(rmse, 6),
                "mae": round(mae, 6),
                "mean_error": round(me, 6),
            },
        )

    result = (
        f"Bias metrics for {var_a} over {desc}: RMSE={rmse:.6f}, "
        f"MAE={mae:.6f}, ME={me:.6f}, std_err={std_err:.6f}, "
        f"max_abs_err={max_abs_err:.6f}, Pearson r={pearson_r:.4f}, "
        f"n={int(mask.sum())}."
    )
    if out_path:
        result += f" Bias map saved to {out_path}."
    return result


def fit_precip_distribution(
    dataset_path: str,
    variable: str | None = None,
    region_bbox: list[float] | None = None,
    point: list[float] | None = None,
    distribution: str = "gamma",
    output_dir: str | None = None,
) -> str:
    """Fit a statistical distribution to a precipitation sample.

    Parameters
    ----------
    dataset_path
        Path to the NetCDF file.
    variable
        Variable name. If None, auto-detected.
    region_bbox
        Bounding box to sample from a region.
    point
        [lat, lon] to sample at a single grid cell. Overrides region_bbox.
    distribution
        One of "gamma", "weibull_min", "gumbel_r", "norm", "lognorm", "expon".
    output_dir
        Optional directory to save a histogram + fitted-PDF figure.

    Returns
    -------
    str
        Estimated parameters, KS test p-value, AIC/BIC and figure path.
    """
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        valid = ", ".join(SUPPORTED_DISTRIBUTIONS.keys())
        raise ValueError(f"distribution must be one of {valid}, got {distribution}")

    da, var, desc = _resolve_dataarray(dataset_path, variable, region_bbox, point)
    if "time" in da.dims and ("lat" in da.dims or "lon" in da.dims):
        # Flatten spatio-temporal sample
        da = da.stack(sample=[d for d in da.dims if d in ("time", "lat", "lon")]).dropna("sample")
    elif "lat" in da.dims and "lon" in da.dims:
        da = da.stack(sample=["lat", "lon"]).dropna("sample")

    sample = da.values.astype(float)
    sample = sample[np.isfinite(sample)]

    if distribution == "expon" or distribution == "gamma":
        # Remove zero values for these strictly positive distributions
        sample = sample[sample > 0]

    if len(sample) < 5:
        return f"Cannot fit {distribution} to {var} at {desc}: fewer than 5 valid values."

    dist = SUPPORTED_DISTRIBUTIONS[distribution]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = dist.fit(sample)

    # KS goodness-of-fit
    ks_stat, ks_pvalue = st.kstest(sample, lambda x: dist.cdf(x, *params))

    # Log-likelihood, AIC and BIC
    logpdf = dist.logpdf(sample, *params)
    log_likelihood = float(np.sum(logpdf[np.isfinite(logpdf)]))
    k = len(params)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(len(sample)) - 2 * log_likelihood

    # Plot histogram + fitted PDF
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sample, bins=min(50, len(sample) // 5 + 1), density=True, alpha=0.6, label="data")
    x = np.linspace(np.min(sample), np.max(sample), 500)
    ax.plot(x, dist.pdf(x, *params), "r-", lw=2, label=f"fitted {distribution}")
    ax.set_title(f"Distribution fit: {distribution} for {var} at {desc}")
    ax.set_xlabel(f"{var}")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()

    out_path = _save_or_return(
        fig,
        f"distfit_{distribution}_{var}_{desc.replace(' ', '_').replace(',', '')}",
        output_dir,
        metadata={
            "variable": var,
            "diagnostic": f"{distribution} distribution fit for {var} at {desc}",
            "distribution": distribution,
            "params": [float(p) for p in params],
            "ks_pvalue": float(ks_pvalue),
            "aic": float(aic),
            "bic": float(bic),
            "n": len(sample),
        },
    )

    return (
        f"Fitted {distribution} distribution to {var} at {desc}: "
        f"params={[float(p) for p in params]}, "
        f"KS p-value={ks_pvalue:.4f}, AIC={aic:.2f}, BIC={bic:.2f}, n={len(sample)}. "
        f"Figure saved to {out_path}."
    )
