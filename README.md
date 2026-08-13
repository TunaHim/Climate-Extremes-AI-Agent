# Climate DT Precipitation Extremes Analysis

AI-assisted diagnostic workflow for analysing precipitation extremes in DestinE Climate DT simulations.

This project is built as a demonstrator for a scientist position in **Climate Data Analysis and AI Methods** for the EU **Destination Earth** Climate Digital Twin initiative.

## Scientific question

What is the added value of kilometre-scale Climate DT simulations (IFS-FESOM, IFS-NEMO, ICON) for representing precipitation extremes compared to ERA5 reanalysis?

## Project structure

```
ClimateDT_PrecipExtremes/
├── assets/                  # Demo figures shipped with the repo for Streamlit Cloud
├── data/                    # Downloaded GRIB/NetCDF files (gitignored)
│   └── demo/                # Small demo datasets (committed, < 25 MB total)
├── figures/                 # Generated plots and sidecar metadata (gitignored)
├── src/                     # Reusable Python modules
│   ├── diagnostics/         # Precipitation/extreme indices
│   ├── plotting/            # Map and time-series plots
│   ├── utils/               # Regridding, I/O helpers
│   └── ai/                  # AI interpretation and agentic tools
├── notebooks/               # Exploratory analysis
├── dashboard/               # Streamlit app
├── scripts/                 # Download and batch processing scripts
├── tests/                   # Unit tests
├── environment.yml          # Conda environment
├── requirements.txt         # Pip requirements
└── README.md
```

## Quick start

### 1. Install environment

```bash
conda env create -f environment.yml
conda activate climate-dt-precip
```

### 2. Download data

#### DestinE Climate DT (IFS-FESOM precipitation)

```bash
python scripts/download_destine_precip.py
```

#### ERA5 reference

```bash
python scripts/download_era5_precip.py
```

### 3. Run diagnostics

```bash
python scripts/run_precip_diagnostics.py
```

### 4. Launch dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## Features

- Reproducible xarray/Dask workflow for GRIB/NetCDF/Zarr data
- Computation of precipitation extreme indices (RX1day, RX5day, R95p, PRCPTOT)
- Bias maps of Climate DT vs ERA5
- Interactive Streamlit dashboard with figure browsing
- AI-assisted figure interpretation (Gemini / local Ollama)
- **Native Gemini function-calling agent**: ask a scientific question, the model calls diagnostic tools and returns a summary with generated maps
- **Deployment-ready demo**: small committed datasets and pre-generated figures for Streamlit Community Cloud

## Towards agentic climate AI

The current dashboard is AI-assisted. The `src/ai/agent.py` module demonstrates an agentic layer where a language model can:

1. Parse a user question (e.g. "Compare 2023 summer precipitation extremes over Europe")
2. Select appropriate diagnostic tools
3. Trigger data loading, computation and plotting
4. Summarise findings

This is a prototype step towards fully autonomous climate diagnostic agents.

## Deployment (Streamlit Community Cloud)

The repository is configured for a free public deployment so you can share a clickable link in your CV.

### What ships with the repo

The following lightweight assets are committed under version control (total ~15 MB):

| File | Size | Purpose |
|---|---|---|
| `data/demo/cpc_south_asia_2013.nc` | ~1.9 MB | Daily CPC subset for live agent extremes (South Asia) |
| `data/demo/cpc_europe_2013.nc` | ~4.2 MB | Daily CPC subset for live agent extremes (Europe) |
| `data/demo/cmip6_global_monthly_2013.nc` | ~3.6 MB | CMIP6 monthly means for global maps |
| `data/demo/cpc_global_monthly_2013_regridded.nc` | ~3.6 MB | CPC monthly means on the CMIP6 grid |
| `assets/demo_figures/*.png` | ~2 MB | Pre-generated mean, bias, and extreme-index maps |

### Deploy steps

1. **Push to GitHub**

   ```bash
   git init
   git add .
   git commit -m "Initial deployment-ready version"
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

   Note: `.gitignore` already excludes the heavy raw data files (`data/cmip6/`, `data/cpc/`, `figures/`) and API secrets. Only the small `data/demo/` and `assets/` files are committed.

2. **Add your Gemini API key**

   Copy the template:

   ```bash
   cp dashboard/.streamlit/secrets.toml.example dashboard/.streamlit/secrets.toml
   ```

   Edit `dashboard/.streamlit/secrets.toml` and replace `your-gemini-api-key` with a real key.

3. **Deploy on Streamlit Cloud**

   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Connect your GitHub account and select the repository.
   - Set the main file path to `dashboard/streamlit_app.py`.
   - In the app settings, add `GEMINI_API_KEY` as a secret.
   - Deploy.

4. **Share the link**

   Once deployed you will get a URL like:

   ```text
   https://<your-repo-name>.streamlit.app
   ```

   Put this link at the top of your CV and cover letter.

### Local run

```bash
conda activate climate-dt-precip
streamlit run dashboard/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

## License

Research use only.

## Contact

For questions or collaboration see the GitHub repository.
