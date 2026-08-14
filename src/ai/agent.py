"""Native Gemini function-calling ReAct agent for climate diagnostics.

The agent receives a natural-language scientific question, calls the Gemini
API with a set of registered Python tools, executes any returned function
calls against local NetCDF files, and returns a concise summary together with
the path to any figure that was generated.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load project root .env (optional)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Import the tool functions and demo configuration that Gemini can use
from src.ai.tools import DEMO_REGIONS, compute_and_plot_extreme


BASE_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = BASE_DIR / "figures"


def _get_gemini_api_key() -> str | None:
    """Return the Gemini API key from env, project secrets.toml, or Streamlit secrets."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key

    # Try the project-level Streamlit secrets file directly (useful in scripts)
    secrets_file = BASE_DIR / "dashboard" / ".streamlit" / "secrets.toml"
    if secrets_file.exists():
        try:
            import tomllib

            with secrets_file.open("rb") as f:
                secrets = tomllib.load(f)
            key = secrets.get("GEMINI_API_KEY", None)
            if key and not key.startswith("your-"):
                return key
        except Exception:
            pass

    # Fall back to Streamlit runtime secrets when running inside the dashboard
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        return None


def _execute_function_call(function_call) -> str:
    """Execute a Gemini FunctionCall and return its result as text."""
    name = function_call.name

    # Convert protobuf Struct to plain JSON-serializable Python types
    try:
        from google.protobuf.json_format import MessageToDict

        args = MessageToDict(function_call.args)
    except Exception:
        args = dict(function_call.args)

    if name == "compute_and_plot_extreme":
        try:
            return compute_and_plot_extreme(**args)
        except Exception as exc:
            return f"Error executing {name}: {exc}"

    return f"Error: Unknown tool '{name}'."


def run_climate_agent(user_query: str, api_key: str | None = None) -> str:
    """Run the Gemini ReAct agent against the user query.

    The model is given the `compute_and_plot_extreme` tool. If it issues a
    function call, the function is executed locally and the result is sent
    back to the model for a final answer.

    Parameters
    ----------
    user_query
        Natural-language scientific question, e.g. "Compute RX1day over
        South Asia [60, 100, 5, 35] using data/cpc/cpc_precip_2013.nc".
    api_key
        Optional Gemini API key. If omitted, reads from GEMINI_API_KEY env var
        or Streamlit secrets.

    Returns
    -------
    str
        Final agent response, including the path to any generated figure.
    """
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is required for the agent. "
            "Install with: pip install google-generativeai"
        ) from exc

    if api_key is None:
        api_key = _get_gemini_api_key()
    if not api_key:
        return (
            "No Gemini API key found. Set GEMINI_API_KEY in your environment "
            "or add it to .streamlit/secrets.toml to run the agent."
        )

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name="gemini-flash-latest",
        tools=[compute_and_plot_extreme],
    )

    demo_hints = "; ".join(
        f"{info['description']} -> {info['path']} with bbox {info['bbox']}"
        for info in DEMO_REGIONS.values()
    )
    system_prompt = (
        "You are an expert climate scientist and AI agent. "
        "Use the available tool to answer the user's question. "
        "When you need a diagnostic, call the tool with the exact arguments. "
        f"Available lightweight demo datasets are: {demo_hints}. "
        "Use one of these if the user does not specify a file path. "
        "Then summarise the result in one or two sentences, including the figure path."
    )

    chat = model.start_chat()
    response = chat.send_message(
        [system_prompt, user_query],
        tool_config={
            "function_calling_config": {
                "mode": "auto",
            }
        },
    )

    # Collect any function calls issued by the model
    function_results = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                result = _execute_function_call(part.function_call)
                function_results.append(result)

    if not function_results:
        # No tool call was made; return whatever the model said
        return response.text

    # Send the function results back to the model for a final answer
    follow_up = chat.send_message(
        "Tool results:\n" + "\n".join(function_results)
    )
    return follow_up.text


class ClimateDiagnosticAgent:
    """Backwards-compatible wrapper around run_climate_agent.

    Existing dashboard code can still instantiate this class with
    data_dir and figures_dir, but the heavy lifting is now done by
    the Gemini function-calling agent.
    """

    def __init__(self, data_dir: Path, figures_dir: Path):
        self.data_dir = data_dir
        self.figures_dir = figures_dir

    def plan(self, question: str, model: str = "gemini") -> list[dict]:
        """Return a single-step plan for the agent."""
        return [
            {
                "tool": "compute_and_plot_extreme",
                "args": {"user_query": question},
                "reason": "Execute the user question via the Gemini ReAct agent",
            }
        ]

    def execute_plan(self, plan: list[dict]) -> dict:
        """Execute the plan by calling the Gemini agent."""
        user_query = plan[0].get("args", {}).get("user_query", "")
        result = run_climate_agent(user_query)
        return {"results": [{"step": plan[0], "result": result}]}
