#!/usr/bin/env python3
"""Streamlit dashboard for Climate DT precipitation extremes analysis."""

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path so absolute src.* imports work everywhere
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.ai.agent import run_climate_agent
from src.ai.interpret import interpret_figure
from src.ai.tools import (
    DEMO_FIGURES_DIR,
    DEMO_REGIONS,
    compute_and_plot_extreme,
    germany_climatology_map,
    global_climatology_map,
)

FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"
DEMO_FIGURES = sorted(DEMO_FIGURES_DIR.glob("*.png")) if DEMO_FIGURES_DIR.exists() else []

INTRO_TEXT = """
This prototype explores precipitation extremes and biases in the **Destination Earth Climate DT**
using a small, curated demo. It compares CMIP6 historical output from **MPI-ESM1-2-HR** against
gauge-based **CPC** observations. Use this page for a quick orientation, then switch to the
**Agentic Demo** tab to ask your own climate-science questions in plain language.
"""

METHODLOGY_TEXT = """
**Datasets**
- **Simulation:** `MPI-ESM1-2-HR` historical CMIP6 daily precipitation (`pr`)
- **Reference:** NOAA CPC Unified Gauge-Based Global Daily Precipitation
- **Period analysed:** 2013 for global/regional maps; 1995–2014 for Germany

**Processing steps**
1. Convert CMIP6 `pr` from kg m⁻² s⁻¹ to mm day⁻¹ (multiply by 86,400).
2. Subset both datasets to the common month or 20-year period.
3. For regional fields, regrid to a common 0.5° regular grid using bilinear interpolation.
4. Compute mean precipitation and ETCCDI-like extreme indices (RX1day, RX5day, R95p).
5. Plot model, reference and bias maps; world maps use a **Robinson projection**.

**Limitations**
Only Germany has a multi-year (1995–2014) monthly record. All other regions are limited to 2013,
so multi-decadal trend and climatology questions are only reliable for Germany.
"""

EXAMPLE_QUESTIONS = [
    "Compare CPC and CMIP6 2013 global mean precipitation and show the bias as a world map.",
    "Compute and compare CPC and CMIP6 20-year mean precipitation over Germany.",
    "Compare observed and modelled precipitation at Frankfurt, Hamburg and Munich from 1995 to 2014.",
    "Is there a significant precipitation trend in Germany between 1995 and 2014, comparing CPC to CMIP6?",
]

AI_MODELS = [
    "Gemini 2.5 Flash",
    "Groq Llama 3.2 Vision",
    "Ministral 3 (Local Ollama)",
    "Gemma 4 Vision (Local Ollama)",
    "Qwen 2.5-VL (Local Ollama)",
]

GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-safeguard-20b",
]


def _list_figures() -> list[Path]:
    """List available PNG figures."""
    if not FIGURES_DIR.exists():
        return []
    return sorted(FIGURES_DIR.glob("*.png"))


def _load_metadata(image_path: Path) -> dict:
    """Load JSON sidecar metadata for a figure."""
    sidecar = image_path.with_suffix(".json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text())
        except Exception:
            pass
    return {}


def _figure_description(metadata: dict, stem: str) -> str:
    """Build a short climate-science caption from figure metadata."""
    if not metadata:
        return f"**{stem.replace('_', ' ').title()}** — no metadata available."

    diagnostic = metadata.get("diagnostic", stem.replace("_", " ").title())
    variable = metadata.get("variable", "precipitation")
    units = metadata.get("units", "mm/day")
    model = metadata.get("model", "")
    models = metadata.get("models", "")
    period = metadata.get("period", "")
    region = metadata.get("region", "")
    index = metadata.get("index", "")

    parts = [f"**{diagnostic}**"]
    if model:
        parts.append(f"Dataset: *{model}*")
    if models:
        parts.append(f"Comparison: *{models}*")
    if variable:
        parts.append(f"Variable: *{variable}*")
    if index:
        parts.append(f"Index: *{index}*")
    if units:
        parts.append(f"Units: *{units}*")
    if region:
        parts.append(f"Region: *{region}*")
    if period:
        parts.append(f"Period: *{period}*")
    return "  \n".join(parts)


def _extract_path_from_message(message: str) -> Path | None:
    """Pull the final file path out of a tool's confirmation message."""
    match = re.search(r"(?:saved (?:it |the |figure )?to\s+)([^\s]+)", message)
    if match:
        return Path(match.group(1))
    return None


@st.cache_data(show_spinner=False)
def _precompute_overview_figures() -> list[Path]:
    """Generate the small set of figures shown on the General analysis tab."""
    paths: list[Path] = []

    # World 3-panel (CPC, CMIP6, bias) in Robinson projection
    msg = global_climatology_map(metric="mean", dataset="both")
    p = _extract_path_from_message(msg)
    if p:
        paths.append(p)

    # Germany 3-panel (CPC, CMIP6, bias) in PlateCarree
    msg = germany_climatology_map(metric="mean", dataset="both")
    p = _extract_path_from_message(msg)
    if p:
        paths.append(p)

    # Regional extreme example: RX1day over Europe
    europe = DEMO_REGIONS["europe"]
    msg = compute_and_plot_extreme(
        dataset_path=str(europe["path"]),
        index_name="RX1day",
        region_bbox=europe["bbox"],
    )
    p = _extract_path_from_message(msg)
    if p:
        paths.append(p)

    return paths


def _data_availability_table() -> pd.DataFrame:
    """Build the table describing every dataset bundled in data/demo/."""
    rows = [
        {
            "File": "cpc_south_asia_2013.nc",
            "Dataset": "CPC (observed)",
            "Region": "South Asia",
            "Years": "2013 (1 yr)",
            "Temporal res.": "Daily",
            "Spatial res.": "~0.5°",
            "Best for": "Extreme index maps (RX1day, RX5day, R95p)",
        },
        {
            "File": "cpc_europe_2013.nc",
            "Dataset": "CPC (observed)",
            "Region": "Europe",
            "Years": "2013 (1 yr)",
            "Temporal res.": "Daily",
            "Spatial res.": "~0.5°",
            "Best for": "Extreme index maps (RX1day, RX5day, R95p)",
        },
        {
            "File": "cmip6_global_monthly_2013.nc",
            "Dataset": "CMIP6 MPI-ESM1-2-HR (model)",
            "Region": "Global",
            "Years": "2013 (1 yr)",
            "Temporal res.": "Monthly",
            "Spatial res.": "~100 km",
            "Best for": "Global mean precipitation maps",
        },
        {
            "File": "cpc_global_monthly_2013_regridded.nc",
            "Dataset": "CPC (observed, regridded)",
            "Region": "Global",
            "Years": "2013 (1 yr)",
            "Temporal res.": "Monthly",
            "Spatial res.": "~100 km (regridded to CMIP6 grid)",
            "Best for": "Model-vs-reference bias maps",
        },
        {
            "File": "cpc_germany_1995_2014_monthly.nc",
            "Dataset": "CPC (observed)",
            "Region": "Germany",
            "Years": "1995-2014 (20 yr)",
            "Temporal res.": "Monthly (gridded)",
            "Spatial res.": "~0.5°",
            "Best for": "20-yr trend, climatology & bias maps",
        },
        {
            "File": "cmip6_germany_1995_2014_monthly.nc",
            "Dataset": "CMIP6 MPI-ESM1-2-HR (model)",
            "Region": "Germany",
            "Years": "1995-2014 (20 yr)",
            "Temporal res.": "Monthly (gridded)",
            "Spatial res.": "~100 km",
            "Best for": "20-yr trend, climatology & bias maps",
        },
    ]
    return pd.DataFrame(rows)


def render_figure_interpretation(fig_path: Path):
    """Add an 'Ask the AI about this figure' expander under a figure."""
    with st.expander("🤖 Ask the AI about this figure"):
        model = st.selectbox(
            "Model",
            AI_MODELS,
            key=f"ai_model_{fig_path.name}",
        )
        prompt = st.text_area(
            "Your question",
            value="Describe the key patterns and biases shown in this figure.",
            key=f"ai_prompt_{fig_path.name}",
        )
        if st.button("Ask the AI", type="primary", key=f"ai_ask_{fig_path.name}"):
            with st.spinner(f"Asking {model} ..."):
                response = interpret_figure(fig_path, prompt, model=model)
            st.markdown(response)


def render_general_analysis():
    """Landing tab: orientation, data inventory and a curated overview gallery."""
    st.header("📊 General analysis")
    st.markdown(INTRO_TEXT)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("📋 Data availability")
        st.markdown(
            "Only Germany has a multi-year record; everything else is limited to 2013. "
            "This table tells you what the agent can realistically answer."
        )
        st.dataframe(_data_availability_table(), use_container_width=True, hide_index=True)
    with col2:
        with st.expander("About / Methodology", expanded=False):
            st.markdown(METHODLOGY_TEXT)

    st.markdown("---")
    st.subheader("🗺️ Overview maps")
    st.markdown(
        "A few pre-computed diagnostics to set the scene. World maps are shown in "
        "**Robinson projection**; regional maps use a local PlateCarree projection."
    )

    overview_paths = _precompute_overview_figures()
    if not overview_paths:
        st.warning("No overview figures could be generated. Check that data/demo/ is populated.")
        return

    for fig_path in overview_paths:
        if not fig_path.exists():
            continue
        metadata = _load_metadata(fig_path)
        title = metadata.get("diagnostic", fig_path.stem.replace("_", " ").title())
        st.markdown(f"### {title}")
        st.image(str(fig_path), use_container_width=True)
        st.markdown(_figure_description(metadata, fig_path.stem))
        render_figure_interpretation(fig_path)
        st.markdown("---")


def _render_figure_grid(figure_paths: list[Path]):
    """Render a set of figures in a responsive grid, each with a download button."""
    n_cols = min(3, len(figure_paths)) or 1
    cols = st.columns(n_cols)
    for i, fig_path in enumerate(figure_paths):
        col = cols[i % n_cols]
        with col:
            metadata = _load_metadata(fig_path)
            caption = metadata.get("diagnostic", fig_path.stem)
            st.image(str(fig_path), caption=caption, use_container_width=True)

            trend = metadata.get("trend_mm_per_day_per_year")
            if trend:
                dataset_labels = {"cpc": "CPC (observed)", "cmip6": "CMIP6 (model)"}
                for key, slope in trend.items():
                    st.metric(f"Trend — {dataset_labels.get(key, key)}", f"{slope:+.4f} mm/day/yr")

            st.download_button(
                "⬇️ Download PNG",
                data=fig_path.read_bytes(),
                file_name=fig_path.name,
                mime="image/png",
                key=f"dl_{fig_path.name}_{i}",
            )


def render_agentic_demo():
    """Live multi-provider function-calling agent that executes climate tools."""
    st.header("🧠 Agentic Diagnostic Execution")
    st.markdown(
        "Ask any climate-science question in plain language. The agent decides "
        "which diagnostic(s) to run and how many figures to produce. The only real "
        "limits are the spatial and temporal extent of the underlying data."
    )

    if "agent_question" not in st.session_state:
        st.session_state["agent_question"] = EXAMPLE_QUESTIONS[0]
    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []

    st.markdown("**Try an example, or write your own question below:**")
    example_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(example_cols, EXAMPLE_QUESTIONS):
        if col.button(example, use_container_width=True):
            st.session_state["agent_question"] = example

    question = st.text_area(
        "Your question",
        key="agent_question",
        height=80,
    )

    st.markdown("**Choose the provider and run it:**")
    tab_gemini, tab_groq = st.tabs(["🚀 Gemini", "⚡ Groq"])

    def _run_agent(provider: str, button_key: str, groq_model: str | None = None) -> None:
        if st.button(f"Run Agent with {provider.title()}", type="primary", key=button_key):
            run_start = time.time()
            with st.spinner(f"Running {provider.title()} agent ..."):
                try:
                    kwargs = {}
                    if groq_model:
                        kwargs["groq_model"] = groq_model
                    result = run_climate_agent(question, provider=provider, **kwargs)
                except Exception as exc:
                    st.error(f"Agent failed: {exc}")
                    return

            agent_figures = sorted(
                (p for p in FIGURES_DIR.glob("agent_*_output.png") if p.stat().st_mtime >= run_start),
                key=lambda p: p.stat().st_mtime,
            )

            st.session_state["agent_history"].insert(
                0,
                {
                    "question": question,
                    "provider": provider,
                    "result": result,
                    "figures": [str(p) for p in agent_figures],
                },
            )

    with tab_gemini:
        _run_agent("gemini", button_key="run_agent_gemini")

    with tab_groq:
        st.markdown("Groq model")
        groq_model = st.selectbox(
            "Select a model",
            GROQ_MODELS,
            index=0,
            label_visibility="collapsed",
            key="groq_model",
        )
        st.caption(
            "Only these models have tool-calling support on this Groq account. "
            "If one fails, try another."
        )
        _run_agent("groq", button_key="run_agent_groq", groq_model=groq_model)

    # Render the most recent run prominently, older runs in a history panel below
    history = st.session_state["agent_history"]
    if history:
        latest_run = history[0]
        st.subheader("Agent result")
        st.caption(f"Provider: {latest_run.get('provider', 'gemini').title()}")
        st.markdown(latest_run["result"])

        figure_paths = [Path(p) for p in latest_run["figures"] if Path(p).exists()]
        if figure_paths:
            st.subheader(f"Generated figure{'s' if len(figure_paths) > 1 else ''}")
            _render_figure_grid(figure_paths)
        else:
            st.info("No agent figure was generated.")

    if len(history) > 1:
        with st.expander(f"🕘 Run history ({len(history) - 1} earlier run(s))"):
            for i, run in enumerate(history[1:], start=1):
                st.markdown(f"**Q{i}: {run['question']}**")
                st.caption(f"Provider: {run.get('provider', 'gemini').title()}")
                st.markdown(run["result"])
                figure_paths = [Path(p) for p in run["figures"] if Path(p).exists()]
                if figure_paths:
                    _render_figure_grid(figure_paths)
                st.markdown("---")


def main():
    st.set_page_config(
        page_title="Precipitation Extremes Analysis",
        layout="wide",
    )

    st.title("Precipitation Extremes — AI Diagnostic Dashboard")
    st.caption("AI-assisted diagnostics for precipitation extremes")

    tab_analysis, tab_agent = st.tabs([
        "📊 General analysis",
        "🧠 Agentic Demo",
    ])

    with tab_analysis:
        render_general_analysis()

    with tab_agent:
        render_agentic_demo()


if __name__ == "__main__":
    main()
