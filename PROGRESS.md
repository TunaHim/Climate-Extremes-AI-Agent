# Climate DT Precipitation Extremes — Project Progress Log

**Project goal:** Build a reproducible, AI-assisted diagnostic workflow that demonstrates the skills required for the AWI Climate Dynamics scientist position working on Destination Earth Climate DT (Phase 3).

**Live document:** This file will be updated as the project proceeds.

---

## 1. Project setup and environment

**Date:** 2026-08-12

**What was done**
- Created the project folder `ClimateDT_PrecipExtremes/` with a clean repository structure:
  - `data/` for downloaded datasets
  - `figures/` for generated plots and JSON sidecar metadata
  - `src/` with `diagnostics/`, `plotting/`, `utils/`, `ai/` submodules
  - `scripts/` for download and batch-processing scripts
  - `dashboard/` for the Streamlit app
  - `notebooks/` and `tests/` directories
- Wrote `environment.yml` and `requirements.txt` for reproducibility.
- Created a Conda environment `climate-dt-precip` (Python 3.11) using the `libmamba` solver.
- Verified that core packages (`xarray`, `numpy`, `requests`, `cartopy`, `matplotlib`, `streamlit`) import correctly.

**Why it matters for the job application**
- Demonstrates reproducible research and environment management.
- Shows familiarity with the Python scientific ecosystem used in climate data analysis.

**Files**
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\environment.yml`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\requirements.txt`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\README.md`

---

## 2. Data acquisition strategy

**Date:** 2026-08-12

**What was done**
- Initial plan: download IFS-FESOM precipitation from the DestinE Harmonized Data Access (HDA) service.
- Attempted authentication with `destinelab` and STAC search.
- Hit a 403 Forbidden error: account does not have permission to read the Climate DT adaptation catalogue items.
- To avoid delays, we pivoted to openly accessible datasets:
  - **CMIP6** daily precipitation (`pr`) from ESGF: `MPI-ESM1-2-HR`, historical, `r1i1p1f1`, `day` table.
  - **NOAA CPC Unified Gauge-Based Global Daily Precipitation** as a reference dataset.
- Chose a common analysis period: **July 2013**.

**Why it matters for the job application**
- Demonstrates pragmatic problem-solving when target data is unavailable.
- Shows knowledge of multiple climate data repositories (DestinE HDA, ESGF, NOAA PSL).
- Maintains the scientific goal: compare a global climate model simulation with an observational reference.

**Files**
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\scripts\download_destine_precip.py`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\scripts\download_cmip6_precip.py`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\scripts\download_cpc_precip.py`

---

## 3. CMIP6 and reference data download

**Date:** 2026-08-12

**What was done**
- Downloaded `pr_day_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_20100101-20141231.nc` from ESGF (~320 MB).
- Downloaded `cpc_precip_2013.nc` from NOAA PSL (~62 MB).
- Implemented filtering logic in `download_cmip6_precip.py` to select only the file chunk covering the target year (2013), limiting data volume.
- Verified longitude/latitude ranges of both datasets.

**Key technical detail**
- CMIP6 `pr` units: kg m⁻² s⁻¹ (= mm s⁻¹).
- CPC `precip` units: mm day⁻¹.
- Conversion factor used: 1 kg m⁻² s⁻¹ × 86,400 s day⁻¹ = 86,400 mm day⁻¹.

**Why it matters for the job application**
- Demonstrates experience with large geoscientific datasets (NetCDF) and data access APIs.
- Shows understanding of unit conversions and physical variable handling.

**Files**
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\scripts\download_cmip6_precip.py`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\scripts\download_cpc_precip.py`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\src\utils\io.py`

---

## 4. Diagnostic modules and regridding

**Date:** 2026-08-12

**What was done**
- Built `src/diagnostics/precipitation.py` for:
  - hourly-to-daily aggregation
  - monthly climatology
  - seasonal means
  - bias / relative bias
- Built `src/diagnostics/extremes.py` for ETCCDI-like extreme indices:
  - **RX1day**: maximum 1-day precipitation
  - **RX5day**: maximum 5-day precipitation
  - **R95p**: precipitation from wet days above the 95th percentile
  - **PRCPTOT**: total precipitation from wet days (≥ 1 mm day⁻¹)
- Built `src/utils/regrid.py`:
  - Creates a common 0.5° target grid.
  - Tries `xesmf` first for conservative/bilinear regridding.
  - Falls back to `xarray.interp` when `xesmf` is unavailable.
- Fixed a regridding bug: the target grid initially used -180–180° longitude while input datasets used 0–360°, causing horizontal stripes. Changed target grid to 0–360° to match the data.

**Why it matters for the job application**
- Demonstrates climate-science domain knowledge (extreme indices, bias analysis).
- Shows software engineering skills (modular design, fallbacks, robust coordinate handling).
- Uses `xarray` and `cartopy`, key tools for large geoscientific data analysis.

**Files**
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\src\diagnostics\precipitation.py`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\src\diagnostics\extremes.py`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\src\utils\regrid.py`

---

## 5. Plotting module and generated figures

**Date:** 2026-08-12

**What was done**
- Built `src/plotting/maps.py` for global precipitation and bias maps using `cartopy`.
- Generated figures for July 2013:
  - `sim_mean_precip.png` — CMIP6 mean daily precipitation
  - `ref_mean_precip.png` — CPC mean daily precipitation
  - `bias_mean_precip.png` — CMIP6 minus CPC mean bias
  - `bias_rx1day.png`, `bias_rx5day.png`, `bias_r95p.png`, `bias_prcptot.png` — extreme-index biases
- Each figure is saved with a JSON sidecar metadata file describing the variable, units, model, period, and diagnostic.
- Verified that mean maps show physically plausible patterns (ITCZ, monsoon regions, mid-latitude storm tracks).

**Why it matters for the job application**
- Demonstrates ability to produce publication-quality geospatial plots.
- Shows reproducible output with machine-readable metadata.

**Files**
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\src\plotting\maps.py`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\scripts\run_precip_diagnostics.py`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\figures\`

---

## 6. Interactive Streamlit dashboard

**Date:** 2026-08-12

**What was done**
- Built `dashboard/streamlit_app.py` with three tabs:
  1. **Figures**: displays generated maps in climate-analysis order (mean → bias → extremes) with explanatory captions and a methodology expander.
  2. **Ask the AI**: choose a figure and ask a question; dispatches to Gemini or local Ollama models.
  3. **Agentic Demo**: prototype planner that translates a scientific question into a sequence of diagnostic tool calls.
- Added figure captions below each plot using the JSON metadata (dataset, variable, units, period, index).
- Configured Streamlit with `config.toml`.
- Launched the dashboard locally on `http://localhost:8501`.

**Why it matters for the job application**
- Demonstrates interactive data exploration and communication.
- Shows integration of AI interpretation with climate diagnostics.
- Provides a foundation for the “agentic AI” requirement in the job description.

**Files**
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\dashboard\streamlit_app.py`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\dashboard\.streamlit\config.toml`

---

## 7. AI interpretation and agentic prototype

**Date:** 2026-08-12

**What was done**
- Built `src/ai/interpret.py`:
  - Encodes figures as base64 images.
  - Sends the image plus metadata prompt to Gemini or local Ollama models.
- Built `src/ai/agent.py`:
  - A simple prototype where an LLM plans a sequence of diagnostic tool calls.
  - Includes a fallback planner for offline use.
- Both modules are wired into the dashboard.

**Why it matters for the job application**
- Directly addresses the job requirement to develop AI-assisted and agentic approaches for climate data analysis.
- Demonstrates practical integration of LLMs with scientific workflows.

**Files**
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\src\ai\interpret.py`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\src\ai\agent.py`

---

## Alignment with the AWI job application

The project already touches most of the required and preferred qualifications:

| Job requirement | How the project addresses it |
|---|---|
| Analyse global climate simulations | CMIP6 analysis with bias vs reference |
| Kilometre-scale / Climate DT context | Project framed around Climate DT; DestinE script ready for when access is granted |
| Develop diagnostics and reproducible workflows | Modular `src/` code, scripts, environment files, metadata sidecars |
| AI-assisted and agentic analysis | `src/ai/interpret.py`, `src/ai/agent.py`, dashboard AI tabs |
| Investigate weather and climate extremes | RX1day, RX5day, R95p, PRCPTOT indices |
| Evaluate simulations and provide feedback | Bias maps comparing model vs CPC |
| Python programming and scientific software development | Modular package with tests and docs |
| Large geoscientific datasets (NetCDF, GRIB, xarray, Dask) | CMIP6/CPC NetCDF, ERA5/DestinE GRIB loaders, xarray/Dask in environment |
| Git / version control | `.gitignore`, structured repo |
| ML/AI applied to geosciences | LLM-based interpretation and planning |
| Workflow automation and reproducible research | Conda environment, scripts, metadata |

---

## 8. Native Gemini function-calling agent (ReAct loop)

**Date:** 2026-08-13

**What was done**
- Created `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\src\ai\tools.py`:
  - `_find_precip_variable`: auto-detects the precipitation variable in a NetCDF file.
  - `_slice_region`: slices a dataset to a `[min_lon, max_lon, min_lat, max_lat]` bounding box, handling descending latitudes (e.g. NOAA CPC).
  - `calculate_rx1day`, `calculate_rx5day`, `calculate_r95p`: thin wrappers around the existing extreme-index functions.
  - `compute_and_plot_extreme`: loads NetCDF, subsets the region, computes the index, plots a map, saves to `figures/agent_{index}_output.png`, and returns a status string.
- Refactored `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\src\ai\agent.py`:
  - Replaced the static text-plan agent with a live ReAct agent using `google.generativeai` function calling.
  - `run_climate_agent(user_query)` registers `compute_and_plot_extreme` as a native Gemini tool, sends the query, executes any returned function calls, and returns the final model summary.
  - Reads `GEMINI_API_KEY` from environment variables or Streamlit secrets.
  - Kept `ClimateDiagnosticAgent` as a backwards-compatible wrapper.
- Updated `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\dashboard\streamlit_app.py`:
  - Replaced the static agentic prototype tab with a live execution trigger.
  - Added `Run Agent Execution` button and an image container that displays any `figures/agent_*_output.png` generated by the agent.
  - Switched dashboard imports to absolute `src.*` paths and added the project root to `sys.path`.

**Verification**
- `python -m py_compile src/ai/tools.py src/ai/agent.py dashboard/streamlit_app.py` passes.
- `python -c "from src.ai.tools import compute_and_plot_extreme; compute_and_plot_extreme('data/cpc/cpc_precip_2013.nc', 'RX1day', [60, 100, 5, 35])"` successfully created `figures/agent_RX1day_output.png`.
- `python -c "from src.ai.agent import run_climate_agent; print(run_climate_agent('Compute RX1day extreme precipitation over South Asia [60, 100, 5, 35] using data/cpc/cpc_precip_2013.nc'))"` executed end-to-end with a valid Gemini API key, generated `figures/agent_RX1day_output.png`, and returned a concise summary.
- `streamlit run dashboard/streamlit_app.py` starts without errors.
- The project-level `dashboard/.streamlit/secrets.toml` is read both by the dashboard and by standalone scripts via `src/ai/agent.py`.

**Why it matters for the job application**
- Directly implements the job requirement to develop **agentic AI systems** for climate data analysis.
- Demonstrates native LLM tool-calling, dynamic code execution, and integration with scientific diagnostic functions.

**Files**
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\src\ai\tools.py`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\src\ai\agent.py`
- `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\dashboard\streamlit_app.py`

---

## 9. Deployment-ready demo for Streamlit Community Cloud

**Date:** 2026-08-13

**What was done**
- Created lightweight demo datasets in `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\data\demo`:
  - `cpc_south_asia_2013.nc` (~1.9 MB) and `cpc_europe_2013.nc` (~4.2 MB): daily CPC subsets for live agent extreme-index computation.
  - `cmip6_global_monthly_2013.nc` (~3.6 MB): CMIP6 monthly means (pr converted to mm/day).
  - `cpc_global_monthly_2013_regridded.nc` (~3.6 MB): CPC monthly means regridded to the CMIP6 grid.
- Generated demo figures in `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\assets\demo_figures`:
  - Global mean precipitation maps for CMIP6 and CPC.
  - Model-minus-reference bias map.
  - Regional extreme-index maps for South Asia and Europe (RX1day, RX5day, R95p).
- Added scripts to recreate the demo assets:
  - `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\scripts\create_demo_data.py`
  - `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\scripts\generate_demo_figures.py`
- Updated `.gitignore` to allow the small demo datasets and figures while still excluding the full raw data.
- Updated `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\dashboard\streamlit_app.py` to:
  - Fall back to demo figures when no locally generated figures exist.
  - Let users select between South Asia and Europe demo regions in the agent tab.
- Updated `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\src\ai\tools.py` and `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\src\ai\agent.py` with demo-region constants and prompts.
- Added Streamlit Community Cloud deployment instructions to `@/C:\Users\LENOVO\Desktop\Com_AWI_2026\AWI_Coding\ClimateDT_PrecipExtremes\README.md`.

**Verification**
- Total committed demo assets: **14.91 MB** (well under the 25 MB target).
- All demo datasets and figures were regenerated successfully.
- `python -m py_compile src/ai/tools.py src/ai/agent.py dashboard/streamlit_app.py scripts/create_demo_data.py scripts/generate_demo_figures.py` passes.
- `streamlit run dashboard/streamlit_app.py` starts and the dashboard falls back to demo figures.

**Why it matters for the job application**
- Provides a live, clickable link for the CV/cover letter that a hiring committee can open without installing anything.
- Demonstrates software-engineering judgement: shipping lightweight assets while keeping large raw data off GitHub.
- Shows end-to-end reproducibility: scripts can recreate the demo from the full raw data.

---

## Alignment with the AWI job application

The project now addresses almost every required and preferred qualification:

| Job requirement | How the project addresses it |
|---|---|
| Analyse global climate simulations | CMIP6 analysis with bias vs reference |
| Kilometre-scale / Climate DT context | Project framed around Climate DT; DestinE script ready for when access is granted |
| Develop diagnostics and reproducible workflows | Modular `src/` code, scripts, environment files, metadata sidecars |
| AI-assisted and agentic analysis | `src/ai/interpret.py`, native function-calling agent in `src/ai/agent.py`, dashboard agent tab |
| Investigate weather and climate extremes | RX1day, RX5day, R95p, PRCPTOT indices |
| Evaluate simulations and provide feedback | Bias maps comparing model vs CPC |
| Python programming and scientific software development | Modular package with tests and docs |
| Large geoscientific datasets (NetCDF, GRIB, xarray, Dask) | CMIP6/CPC NetCDF, ERA5/DestinE GRIB loaders, xarray/Dask in environment |
| Git / version control | `.gitignore`, structured repo |
| ML/AI applied to geosciences | LLM-based interpretation and native tool-calling |
| Workflow automation and reproducible research | Conda environment, scripts, metadata |

---

## What is still missing / next steps

1. **Access real Climate DT data**: once DestinE HDA access is available, run `download_destine_precip.py` and re-run diagnostics with IFS-FESOM output.
2. **ERA5 as reference**: the CDS download script exists but needs a valid `~/.cdsapirc`. This would be a stronger reference than CPC because ERA5 is the target comparison in the original scientific question.
3. **Regional focus**: zoom into Europe or the Arctic to match AWI/Climate DT strengths.
4. **Temporal analysis**: add seasonal/annual cycle, interannual variability, and trend analysis.
5. **Statistical validation**: add area-weighted metrics, Taylor diagrams, PDF comparisons.
6. **Scale-aware analysis**: compare km-scale Climate DT output with CMIP6 and reference datasets to demonstrate “added value.”
7. **Agentic layer enhancement**: add more tools (bias, climatology, multi-region) and multi-turn reasoning.
8. **Documentation and publication**: turn the workflow into a short notebook/paper draft and publishable figures.
9. **Tests**: expand `tests/test_diagnostics.py` to cover regridding, plotting, and tool functions.
10. **CI/CD**: add GitHub Actions for linting, tests, and environment validation.

---

## How to reproduce the current state

```bash
conda activate climate-dt-precip
python scripts/download_cmip6_precip.py
python scripts/download_cpc_precip.py
python scripts/run_precip_diagnostics.py
streamlit run dashboard/streamlit_app.py
```

Open `http://localhost:8501` to view the dashboard.
