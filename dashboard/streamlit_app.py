#!/usr/bin/env python3
"""Streamlit dashboard for precipitation extremes analysis."""

import json
import re
import sys
import threading
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
This prototype explores precipitation extremes and biases using a small, curated demo.
It compares CMIP6 historical output from **MPI-ESM1-2-HR** against gauge-based **CPC**
observations. Use this page for a quick orientation, then switch to the **Agentic Demo** tab
to ask your own climate-science questions in plain language.
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

**AI role**
All numerical climate diagnostics are produced by deterministic Python/xarray functions.
The LLM selects tools, passes parameters and interprets the returned results.
Interpretation is not independent scientific evidence and should not be treated as attribution.
"""

EXAMPLE_QUESTIONS = [
    "Compare CPC and CMIP6 2013 global mean precipitation and show the bias as a world map.",
    "Compute and compare CPC and CMIP6 20-year mean precipitation over Germany.",
    "Compare observed and modelled precipitation at Frankfurt, Hamburg and Munich from 1995 to 2014.",
    "Is there a significant precipitation trend in Germany between 1995 and 2014, comparing CPC to CMIP6?",
    "Map CPC RX1day and RX5day over South Asia for 2013 and explain the differences.",
    "What is the spatial pattern correlation between CMIP6 and CPC global precipitation in 2013?",
    "Show the CMIP6 minus CPC bias map for mean precipitation over Europe in 2013.",
    "For Germany 1995–2014, is the CPC precipitation trend statistically different from the CMIP6 trend?",
]

RX1DAY_DEFINITION = """
**RX1day over selected region** — annual maximum 1-day precipitation amount recorded in each grid cell.
- **Variable:** precipitation
- **Index:** RX1day
- **Units:** mm
"""

AI_MODELS = [
    "Gemini 2.5 Flash",
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


def _render_analysis_info(fig_path: Path):
    """Show the structured provenance metadata associated with a figure."""
    metadata = _load_metadata(fig_path)
    if not metadata:
        return
    with st.expander("📋 Analysis information"):
        st.json(metadata, expanded=False)


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

    # Regional extreme example: RX1day over Europe and South Asia
    for region_name in ["Europe", "South Asia"]:
        region = DEMO_REGIONS[region_name.lower().replace(" ", "_")]
        msg = compute_and_plot_extreme(
            dataset_path=str(region["path"]),
            index_name="RX1day",
            region_bbox=region["bbox"],
            region_name=region_name,
        )
        p = _extract_path_from_message(msg)
        if p:
            paths.append(p)

    return paths


def _data_availability_table() -> pd.DataFrame:
    """Build the table describing every dataset bundled in data/demo/."""
    rows = [
        {
            "Region": "Germany — CPC (obs)",
            "Years": "1995-2014 (20 yr)",
            "Temporal res.": "Monthly (gridded)",
            "Spatial res.": "~0.5°",
            "Best for": "20-yr trend, climatology & bias maps",
            "File": "cpc_germany_1995_2014_monthly.nc",
        },
        {
            "Region": "Germany — CMIP6 MPI-ESM1-2-HR",
            "Years": "1995-2014 (20 yr)",
            "Temporal res.": "Monthly (gridded)",
            "Spatial res.": "~100 km",
            "Best for": "20-yr trend, climatology & bias maps",
            "File": "cmip6_germany_1995_2014_monthly.nc",
        },
        {
            "Region": "Global — CPC (obs, regridded)",
            "Years": "2013 (1 yr)",
            "Temporal res.": "Monthly",
            "Spatial res.": "~100 km (regridded to CMIP6 grid)",
            "Best for": "Model-vs-reference bias maps",
            "File": "cpc_global_monthly_2013_regridded.nc",
        },
        {
            "Region": "Global — CMIP6 MPI-ESM1-2-HR",
            "Years": "2013 (1 yr)",
            "Temporal res.": "Monthly",
            "Spatial res.": "~100 km",
            "Best for": "Global mean precipitation maps",
            "File": "cmip6_global_monthly_2013.nc",
        },
        {
            "Region": "South Asia — CPC (obs)",
            "Years": "2013 (1 yr)",
            "Temporal res.": "Daily",
            "Spatial res.": "~0.5°",
            "Best for": "Extreme index maps (RX1day, RX5day, R95p)",
            "File": "cpc_south_asia_2013.nc",
        },
        {
            "Region": "Europe — CPC (obs)",
            "Years": "2013 (1 yr)",
            "Temporal res.": "Daily",
            "Spatial res.": "~0.5°",
            "Best for": "Extreme index maps (RX1day, RX5day, R95p)",
            "File": "cpc_europe_2013.nc",
        },
    ]
    return pd.DataFrame(rows)


def _right_panel_data_table() -> pd.DataFrame:
    """Compact two-column data availability summary for the agent tab."""
    rows = [
        {"Region / Dataset": "Germany — CPC (obs)", "Coverage": "1995–2014 (20 yr) — Monthly"},
        {"Region / Dataset": "Germany — CMIP6 MPI-ESM1-2-HR", "Coverage": "1995–2014 (20 yr) — Monthly"},
        {"Region / Dataset": "Global — CPC (obs, regridded)", "Coverage": "2013 (1 yr) — Monthly"},
        {"Region / Dataset": "Global — CMIP6 MPI-ESM1-2-HR", "Coverage": "2013 (1 yr) — Monthly"},
        {"Region / Dataset": "South Asia — CPC (obs)", "Coverage": "2013 (1 yr) — Daily"},
        {"Region / Dataset": "Europe — CPC (obs)", "Coverage": "2013 (1 yr) — Daily"},
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

    clim_paths = overview_paths[:2]
    rx1day_paths = overview_paths[2:]

    # Climatology figures (global and Germany 3-panel maps)
    for fig_path in clim_paths:
        if not fig_path.exists():
            continue
        metadata = _load_metadata(fig_path)
        title = metadata.get("diagnostic", fig_path.stem.replace("_", " ").title())
        st.markdown(f"### {title}")
        st.image(str(fig_path), use_container_width=True)
        st.markdown(_figure_description(metadata, fig_path.stem))
        _render_analysis_info(fig_path)
        render_figure_interpretation(fig_path)
        st.markdown("---")

    # RX1day over Europe and South Asia, shown side by side
    st.markdown("### RX1day over selected region")
    cols = st.columns(2)
    for i, fig_path in enumerate(rx1day_paths):
        if not fig_path.exists():
            continue
        metadata = _load_metadata(fig_path)
        region = metadata.get("region", "selected region")
        with cols[i]:
            st.image(str(fig_path), use_container_width=True)
            st.caption(f"RX1day — {region}")
    st.markdown(RX1DAY_DEFINITION)
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
            _render_analysis_info(fig_path)

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


def _render_chat_figures(figure_paths: list[Path]):
    """Render agent-generated figures inside a chat message."""
    for i, fig_path in enumerate(figure_paths):
        if not fig_path.exists():
            continue
        metadata = _load_metadata(fig_path)
        caption = metadata.get("diagnostic", fig_path.stem.replace("_", " ").title())
        st.image(str(fig_path), caption=caption, use_container_width=True)
        _render_analysis_info(fig_path)

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
            key=f"chat_dl_{fig_path.name}_{i}",
        )


class AgentTask:
    """Run run_climate_agent in a background thread so the UI can show a stop button."""

    def __init__(self, question: str, provider: str, groq_model: str | None):
        self.question = question
        self.provider = provider
        self.groq_model = groq_model
        self.stop_event = threading.Event()
        self.result: dict | None = None
        self.error: Exception | None = None
        self._thread = threading.Thread(target=self._target, daemon=True)

    def _target(self) -> None:
        try:
            self.result = run_climate_agent(
                self.question,
                provider=self.provider,
                return_structured=True,
                should_stop=self.stop_event.is_set,
                groq_model=self.groq_model,
            )
        except Exception as exc:
            self.error = exc

    def start(self) -> None:
        self._thread.start()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()


def _finalize_agent_task(run_start: float) -> None:
    """Collect a finished AgentTask from session state and append its result to history."""
    task = st.session_state.get("agent_task")
    if task is None:
        return
    st.session_state["agent_task"] = None

    if task.error:
        summary = f"Agent failed: {task.error}"
        activity = []
    else:
        summary = (task.result or {}).get("summary", "")
        activity = (task.result or {}).get("activity", [])

    agent_figures = sorted(
        (p for p in FIGURES_DIR.glob("agent_*_output.png") if p.stat().st_mtime >= run_start),
        key=lambda p: p.stat().st_mtime,
    )

    st.session_state["agent_history"].append(
        {
            "question": task.question,
            "provider": task.provider,
            "summary": summary,
            "activity": activity,
            "figures": [str(p) for p in agent_figures],
        }
    )


def render_agentic_demo():
    """Live multi-provider function-calling agent in a VS Code-style chat layout."""
    st.header("🧠 Agentic Diagnostic Execution")
    st.caption(
        "All numerical diagnostics are computed by deterministic Python/xarray functions. "
        "The LLM selects tools and interprets the returned results; interpretation is not "
        "independent scientific evidence."
    )

    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []
    if "pending_query" not in st.session_state:
        st.session_state["pending_query"] = ""
    if "agent_task" not in st.session_state:
        st.session_state["agent_task"] = None
    if "agent_task_start" not in st.session_state:
        st.session_state["agent_task_start"] = 0.0
    if "agent_question" not in st.session_state:
        st.session_state["agent_question"] = ""
    if "clear_input" not in st.session_state:
        st.session_state["clear_input"] = False

    # Clear the text input from a previous Send click before the widget is rendered
    if st.session_state.get("clear_input"):
        st.session_state["agent_question"] = ""
        st.session_state["clear_input"] = False

    # Provider / model selector at the top
    provider_col, model_col = st.columns([1, 2])
    with provider_col:
        provider = st.radio(
            "Provider",
            ["Gemini", "Groq"],
            horizontal=True,
            key="agent_provider",
        )
    with model_col:
        groq_model = None
        if provider == "Groq":
            groq_model = st.selectbox(
                "Groq model",
                GROQ_MODELS,
                index=0,
                label_visibility="collapsed",
                key="agent_groq_model",
            )
        else:
            st.caption("Gemini Flash (latest)")

    st.markdown("---")

    # Start a pending query in a background thread
    pending_query = st.session_state.get("pending_query", "")
    active_task = st.session_state.get("agent_task")
    if pending_query and not (active_task and active_task.is_alive):
        st.session_state["pending_query"] = ""
        task = AgentTask(pending_query, provider.lower(), groq_model)
        st.session_state["agent_task"] = task
        st.session_state["agent_task_start"] = time.time()
        task.start()
        st.rerun()

    col_main, col_side = st.columns([3, 1])

    with col_side:
        st.subheader("📋 Data availability")
        st.markdown("Only Germany has a multi-year record.")
        st.dataframe(
            _right_panel_data_table(),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.markdown("**Try an example:**")
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            if st.button(
                example,
                use_container_width=True,
                key=f"agent_example_{i}",
            ):
                st.session_state["agent_question"] = example
                st.rerun()

    with col_main:
        # Conversation history
        for run in st.session_state["agent_history"]:
            with st.chat_message("user"):
                st.markdown(run["question"])
            with st.chat_message("assistant"):
                st.markdown(run.get("summary", ""))
                if run.get("activity"):
                    with st.expander("Agent activity"):
                        for step in run["activity"]:
                            st.markdown(f"- **{step['action']}**: {step['detail']}")
                figure_paths = [Path(p) for p in run.get("figures", []) if Path(p).exists()]
                if figure_paths:
                    _render_chat_figures(figure_paths)

        # Chat input + Send/Stop at the bottom of the main conversation column
        input_col, action_col = st.columns([3, 1])
        with input_col:
            st.text_area(
                "Your question",
                key="agent_question",
                height=80,
                placeholder="Ask a climate-science question...",
                label_visibility="collapsed",
            )
        with action_col:
            send_clicked = st.button(
                "Send",
                type="primary",
                use_container_width=True,
                key="agent_send",
            )
            task = st.session_state.get("agent_task")
            if task and task.is_alive:
                if st.button(
                    "⏹ Stop",
                    use_container_width=True,
                    key="agent_stop",
                ):
                    task.stop_event.set()
                    st.rerun()
                st.info(f"Running {task.provider.title()} agent ...")

        if send_clicked:
            query = st.session_state.get("agent_question", "").strip()
            if query:
                st.session_state["pending_query"] = query
                st.session_state["clear_input"] = True
                st.rerun()

    # Poll while the agent task is running, and finalize it when it finishes
    task = st.session_state.get("agent_task")
    if task and task.is_alive:
        time.sleep(0.5)
        st.rerun()
    elif task is not None and not task.is_alive:
        _finalize_agent_task(st.session_state.get("agent_task_start", time.time()))
        st.rerun()


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
