# Precipitation Extremes Analysis

Live demo: https://climate-extremes-ai-agent.streamlit.app/

AI-assisted diagnostic workflow for analysing precipitation extremes. The dashboard puts an agentic AI layer in front of reproducible climate diagnostics: ask a question in plain language and a multi-provider LLM selects, runs and summarises the right tools.

## What the agent can do

The `src/ai/agent.py` module implements a ReAct-style function-calling agent. Given a scientific question, the model can:

1. Choose from a registry of diagnostic tools (extreme indices, climatology, bias, trends, point comparisons, distribution fits).
2. Call the selected tools, which load demo data, compute diagnostics and save figures.
3. Observe the outputs and produce a final summary with figure paths.

Supported LLM providers:

- **Gemini** (Google)
- **Groq** (OpenAI-compatible endpoint)
- **Ollama** for local figure interpretation

The agents work best on questions that can be answered from the provided demo datasets (extreme indices, climatology, bias, trends, point comparisons). Provider behaviour differs:

- **Gemini** can both run the diagnostic tools and answer broader climate-science questions from its training data, so it may discuss teleconnections such as NAO even though the demo data does not contain those indices.
- **Groq** follows the system instructions more strictly and usually explains data limitations when the requested data is not available.

Neither provider invents computed results for data that is not in the demo datasets.

## Project structure

```
PrecipExtremes/
├── assets/                  # Demo figures shipped with the repo for Streamlit Cloud
├── data/                    # Downloaded NetCDF files (gitignored)
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

### 2. Run diagnostics (optional)

```bash
python scripts/run_precip_diagnostics.py
```

### 3. Launch dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Features

- Reproducible `xarray`/`Dask` workflow for NetCDF data.
- Computation of ETCCDI precipitation extreme indices (RX1day, RX5day, R95p, PRCPTOT).
- Climatology, bias and trend maps (CMIP6 vs CPC observations).
- Automatic projection selection: **Robinson** for global maps, **Plate Carrée** for regional maps.
- Interactive Streamlit dashboard with:
  - **General analysis** tab: data overview, methodology and a curated figure gallery.
  - **Agentic Demo** tab: ask a question and run the multi-provider function-calling agent.
- AI-assisted figure interpretation via **Gemini**; **Ollama** works if a local model is installed and running; **Groq** figure interpretation is currently unavailable because the configured vision model has been decommissioned.
- Multi-provider function-calling agent (Gemini / Groq).
- Deployment-ready demo with small committed datasets and pre-generated figures.

## Demo datasets

The repository ships with lightweight demo data (~15 MB) so the Streamlit app works out of the box:

| File | Size | Purpose |
|---|---|---|
| `data/demo/cpc_south_asia_2013.nc` | ~1.9 MB | Daily CPC subset for live agent extremes (South Asia) |
| `data/demo/cpc_europe_2013.nc` | ~4.2 MB | Daily CPC subset for live agent extremes (Europe) |
| `data/demo/cmip6_global_monthly_2013.nc` | ~3.6 MB | CMIP6 monthly means for global maps |
| `data/demo/cpc_global_monthly_2013_regridded.nc` | ~3.6 MB | CPC monthly means on the CMIP6 grid |
| `assets/demo_figures/*.png` | ~2 MB | Pre-generated mean, bias, and extreme-index maps |

## Deployment (Streamlit Community Cloud)

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial deployment-ready version"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Note: `.gitignore` excludes large raw data files (`data/cmip6/`, `data/cpc/`, `figures/`) and API secrets. Only `data/demo/` and `assets/` are committed.

### 2. Add API keys

Copy the template:

```bash
cp dashboard/.streamlit/secrets.toml.example dashboard/.streamlit/secrets.toml
```

Edit `dashboard/.streamlit/secrets.toml` and add your keys:

- `GEMINI_API_KEY` for the Gemini agent.
- `GROQ_API_KEY` if you want to use the Groq agent.

### 3. Deploy on Streamlit Cloud

- Go to [share.streamlit.io](https://share.streamlit.io).
- Connect your GitHub account and select the repository.
- Set the main file path to `dashboard/streamlit_app.py`.
- Add `GEMINI_API_KEY` and optionally `GROQ_API_KEY` as secrets.
- Deploy.

Once deployed you will get a URL like:

```text
https://<your-repo-name>.streamlit.app
```

## Local run

```bash
conda activate climate-dt-precip
streamlit run dashboard/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

## License

Research use only.

## Contact

For questions or collaboration see the GitHub repository.
