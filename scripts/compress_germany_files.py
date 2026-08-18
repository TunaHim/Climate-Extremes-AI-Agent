"""Re-save the Germany daily/monthly demo NetCDFs with zlib compression
to shrink file size without losing any data."""
from pathlib import Path

import xarray as xr

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = BASE_DIR / "data" / "demo"

files = [
    "cpc_germany_1995_2014_daily.nc",
    "cpc_germany_1995_2014_monthly.nc",
    "cmip6_germany_1995_2014_daily.nc",
    "cmip6_germany_1995_2014_monthly.nc",
]

for fname in files:
    path = DEMO_DIR / fname
    before = path.stat().st_size / 1e6
    ds = xr.open_dataset(path)
    encoding = {var: {"zlib": True, "complevel": 5} for var in ds.data_vars}
    tmp_path = path.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp_path, encoding=encoding)
    ds.close()
    tmp_path.replace(path)
    after = path.stat().st_size / 1e6
    print(f"{fname}: {before:.3f} MB -> {after:.3f} MB")
