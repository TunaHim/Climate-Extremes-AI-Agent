#!/usr/bin/env python3
"""Streamlit dashboard for Climate DT precipitation extremes analysis."""

import json
import sys
from pathlib import Path

import streamlit as st

# Add project root to path so absolute src.* imports work everywhere
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.ai.agent import ClimateDiagnosticAgent
from src.ai.interpret import interpret_figure
from src.ai.tools import DEMO_FIGURES_DIR, DEMO_REGIONS

FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"
DEMO_FIGURES = sorted(DEMO_FIGURES_DIR.glob("*.png")) if DEMO_FIGURES_DIR.exists() else []


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


def render_sidebar():
    """Render settings sidebar."""
    st.sidebar.header("⚙️ Settings")
    st.sidebar.markdown("---")

    st.sidebar.subheader("🤖 AI Model")
    ai_model = st.sidebar.selectbox(
        "Select model",
        options=[
            "Gemini 2.5 Flash",
            "Ministral 3 (Local Ollama)",
            "Gemma 4 Vision (Local Ollama)",
            "Qwen 2.5-VL (Local Ollama)",
        ],
        help="Local Ollama models require Ollama running on your machine.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Agentic Demo")
    enable_agent = st.sidebar.checkbox("Enable agentic planning", value=False)

    return ai_model, enable_agent


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
    if period:
        parts.append(f"Period: *{period}*")
    return "  \n".join(parts)


def render_figure_viewer():
    """Show all generated figures in a climate-analysis order: mean, bias, extremes."""
    st.header("📊 Generated Figures")
    st.markdown(
        "This section presents the diagnostic figures step-by-step. "
        "Start with the mean precipitation maps, then examine the bias, "
        "and finally the precipitation-extreme indices."
    )
    with st.expander("Methodology", expanded=True):
        st.markdown(
            """
            **Datasets**
            - **Simulation:** `MPI-ESM1-2-HR` historical CMIP6 daily precipitation (`pr`)
            - **Reference:** NOAA CPC Unified Gauge-Based Global Daily Precipitation
            - **Period analysed:** July 2013

            **Processing steps**
            1. Convert CMIP6 `pr` from kg m⁻² s⁻¹ to mm day⁻¹ (multiply by 86,400).
            2. Subset both datasets to the common month (July 2013).
            3. Regrid both fields to a common 0.5° regular grid using bilinear interpolation.
            4. Compute mean precipitation and ETCCDI-like extreme indices (RX1day, RX5day, R95p, PRCPTOT).
            5. Plot model, reference, and bias maps for each diagnostic.
            """
        )

    figures = _list_figures()

    # Fallback to lightweight demo figures for Streamlit Cloud / first-time visitors
    if not figures:
        if DEMO_FIGURES:
            st.info("No locally generated figures found. Showing lightweight demo figures that ship with the repository.")
            figures = DEMO_FIGURES
        else:
            st.warning(f"No figures found in `{FIGURES_DIR}` or `{DEMO_FIGURES_DIR}`. Run `python scripts/run_precip_diagnostics.py` first.")
            return

    # Order figures: mean maps first, then mean bias, then extreme biases, then the rest
    mean_figs = [p for p in figures if "mean" in p.stem and "bias" not in p.stem]
    bias_mean_figs = [p for p in figures if "mean" in p.stem and "bias" in p.stem]
    bias_extreme_figs = [p for p in figures if "bias" in p.stem and "mean" not in p.stem]
    other_figs = [p for p in figures if p not in mean_figs + bias_mean_figs + bias_extreme_figs]
    ordered_figs = mean_figs + bias_mean_figs + bias_extreme_figs + other_figs

    for fig_path in ordered_figs:
        metadata = _load_metadata(fig_path)
        title = metadata.get("diagnostic", fig_path.stem.replace("_", " ").title())
        st.subheader(title)
        st.image(str(fig_path), use_container_width=True)
        st.markdown(_figure_description(metadata, fig_path.stem))
        st.markdown("---")


def render_ai_interpretation(ai_model: str):
    """Allow user to select a figure and ask an LLM about it."""
    st.header("🤖 AI Figure Interpretation")

    figures = _list_figures() or DEMO_FIGURES
    if not figures:
        st.warning("No figures available. Generate figures first.")
        return

    # Same ordering as figure viewer: mean maps first, then biases, then the rest
    mean_figs = [p for p in figures if "mean" in p.stem and "bias" not in p.stem]
    bias_mean_figs = [p for p in figures if "mean" in p.stem and "bias" in p.stem]
    bias_extreme_figs = [p for p in figures if "bias" in p.stem and "mean" not in p.stem]
    other_figs = [p for p in figures if p not in mean_figs + bias_mean_figs + bias_extreme_figs]
    ordered_figs = mean_figs + bias_mean_figs + bias_extreme_figs + other_figs

    selected_path = st.selectbox(
        "Select figure",
        ordered_figs,
        format_func=lambda p: _load_metadata(p).get("diagnostic", p.stem),
    )

    if selected_path:
        st.image(str(selected_path), use_container_width=True)
        metadata = _load_metadata(selected_path)
        st.markdown(_figure_description(metadata, selected_path.stem))
        if metadata:
            with st.expander("Raw figure metadata"):
                st.json(metadata)

        prompt = st.text_area(
            "Your question",
            value="Describe the key patterns and biases shown in this figure.",
        )

        if st.button("Ask the AI", type="primary"):
            with st.spinner(f"Asking {ai_model} ..."):
                response = interpret_figure(selected_path, prompt, model=ai_model)
            st.markdown(response)


def render_agentic_demo():
    """Live Gemini function-calling agent that executes climate tools."""
    st.header("🧠 Agentic Diagnostic Execution")
    st.markdown(
        "This agent uses Gemini's native function calling. "
        "Describe what you want (e.g. compute RX1day over a region) and the agent "
        "will run the diagnostic tool and display the generated figure."
    )

    st.subheader("Demo settings")
    demo_region = st.selectbox(
        "Demo region",
        options=list(DEMO_REGIONS.keys()),
        format_func=lambda k: DEMO_REGIONS[k]["description"],
    )
    dataset_path = DEMO_REGIONS[demo_region]["path"]
    region_bbox = DEMO_REGIONS[demo_region]["bbox"]

    question = st.text_input(
        "Ask a scientific question",
        value=f"Compute RX1day extreme precipitation over {DEMO_REGIONS[demo_region]['description']} {region_bbox}",
    )

    final_question = (
        f"{question} using {dataset_path} with region_bbox {region_bbox}. "
        "If no dataset path is given, use the demo dataset."
    )

    if st.button("Run Agent Execution", type="primary"):
        with st.spinner("Running Gemini agent ..."):
            try:
                from src.ai.agent import run_climate_agent
                result = run_climate_agent(final_question)
            except Exception as exc:
                st.error(f"Agent failed: {exc}")
                return

        st.subheader("Agent result")
        st.markdown(result)

        # Display any agent-generated figure
        agent_figures = sorted(FIGURES_DIR.glob("agent_*_output.png"))
        if agent_figures:
            latest = agent_figures[-1]
            st.subheader("Generated figure")
            st.image(str(latest), use_container_width=True)
        else:
            st.info("No agent figure was generated.")


def main():
    st.set_page_config(
        page_title="Climate DT Precipitation Extremes",
        layout="wide",
    )

    st.title("Climate DT — Precipitation Extremes Dashboard")
    st.caption("AI-assisted diagnostics for Destination Earth Climate DT simulations")

    ai_model, enable_agent = render_sidebar()

    tab_figures, tab_ai, tab_agent = st.tabs([
        "📊 Figures",
        "💬 Ask the AI",
        "🧠 Agentic Demo",
    ])

    with tab_figures:
        render_figure_viewer()

    with tab_ai:
        render_ai_interpretation(ai_model)

    with tab_agent:
        if enable_agent:
            render_agentic_demo()
        else:
            st.info("Enable 'Agentic planning' in the sidebar to use this prototype.")


if __name__ == "__main__":
    main()
