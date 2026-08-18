"""Multi-provider function-calling ReAct agent for climate diagnostics.

The agent receives a natural-language scientific question, calls either the
Google Gemini or the Groq (OpenAI-compatible) API with a set of registered
Python tools, executes any returned function calls against local NetCDF files,
and returns a concise summary together with the path(s) to any figures that
were generated.
"""

import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, get_args, get_origin

from dotenv import load_dotenv

# Load project root .env (optional)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Import the tool functions and demo configuration the agent can use
from src.ai.code_sandbox import execute_xarray_script
from src.ai.knowledge import answer_climate_question
from src.ai.stats import (
    bias_metrics,
    fit_precip_distribution,
    linear_regression_analysis,
    spatial_pattern_correlation,
)
from src.ai.tools import (
    DEMO_GERMANY_BBOX,
    DEMO_GERMANY_MONTHLY,
    DEMO_REGIONS,
    compare_precip_at_point,
    compute_and_plot_extreme,
    germany_climatology_map,
    global_climatology_map,
    regional_precip_trend,
)


BASE_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = BASE_DIR / "figures"


def _get_api_key(provider: str) -> str | None:
    """Return the API key for the chosen provider from env, secrets.toml or Streamlit secrets."""
    env_var = f"{provider.upper()}_API_KEY"
    key = os.getenv(env_var)
    if key:
        return key

    # Try the project-level Streamlit secrets file directly (useful in scripts)
    secrets_file = BASE_DIR / "dashboard" / ".streamlit" / "secrets.toml"
    if secrets_file.exists():
        try:
            import tomllib

            with secrets_file.open("rb") as f:
                secrets = tomllib.load(f)
            key = secrets.get(env_var, None)
            if key and not key.startswith("your-"):
                return key
        except Exception:
            pass

    # Fall back to Streamlit runtime secrets when running inside the dashboard
    try:
        import streamlit as st
        return st.secrets.get(env_var, None)
    except Exception:
        return None


def _python_type_to_json_schema(py_type: Any) -> dict:
    """Map a Python type annotation to an OpenAI JSON schema property fragment."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is list or origin is tuple:
        item_type = args[0] if args else Any
        return {"type": "array", "items": _python_type_to_json_schema(item_type)}
    if origin is dict:
        return {"type": "object"}

    # Handle Optional[X] = Union[X, None]
    if origin is type(Any) or py_type is Any:
        return {}

    if origin is not None:
        none_in_args = type(None) in args
        non_none = [a for a in args if a is not type(None)]
        if none_in_args and len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])

    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        Any: {},
        type(None): {},
    }
    return mapping.get(py_type, {})


TOOLS = [
    compute_and_plot_extreme,
    regional_precip_trend,
    germany_climatology_map,
    global_climatology_map,
    compare_precip_at_point,
    linear_regression_analysis,
    spatial_pattern_correlation,
    bias_metrics,
    fit_precip_distribution,
    answer_climate_question,
    execute_xarray_script,
]

TOOL_REGISTRY: dict[str, Callable] = {fn.__name__: fn for fn in TOOLS}


def _parse_param_docs(docstring: str, param_names: list[str]) -> dict[str, str]:
    """Extract one-line descriptions for each parameter from a NumPy-style docstring."""
    param_docs: dict[str, str] = {}
    if not docstring:
        return param_docs

    lines = docstring.splitlines()
    in_params = False
    current_param: str | None = None
    current_desc: list[str] = []

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("parameters"):
            in_params = True
            continue

        if in_params and (
            lower.startswith("returns")
            or lower.startswith("examples")
            or lower.startswith("notes")
            or lower.startswith("see also")
            or lower.startswith("references")
        ):
            if current_param and current_desc:
                param_docs[current_param] = " ".join(current_desc).strip()
            break

        if in_params and stripped.startswith("-" * 3):
            continue

        if in_params:
            # Detect a parameter line: either "name" or "name : type"
            match = None
            for p_name in param_names:
                # Accept "name", "name :", "name -" or "name\n    desc"
                if stripped == p_name or stripped.startswith(p_name + " "):
                    # commit previous param
                    if current_param and current_desc:
                        param_docs[current_param] = " ".join(current_desc).strip()
                    current_param = p_name
                    current_desc = []
                    # If description follows on the same line after " - " or " : ", capture it
                    rest = stripped[len(p_name):].strip()
                    for sep in (" - ", " : ", "--", "—"):
                        if sep in rest:
                            current_desc.append(rest.split(sep, 1)[1].strip())
                            break
                    break
            else:
                # Not a parameter header: continuation line for the current param
                if current_param is not None and stripped:
                    current_desc.append(stripped)

    if current_param and current_desc:
        param_docs[current_param] = " ".join(current_desc).strip()

    return param_docs


def _build_tool_schemas(tools: list[Callable]) -> list[dict]:
    """Build OpenAI/Groq-compatible tool definitions from Python functions."""
    schemas = []
    for fn in tools:
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            schema = _python_type_to_json_schema(param.annotation)
            description = None
            if param.default is not inspect.Parameter.empty:
                schema["default"] = param.default
            else:
                required.append(param_name)
            if schema.get("type") == "string" and isinstance(param.default, str) and param.default:
                # expose common defaults as enum-ish guidance, not a hard enum
                pass
            properties[param_name] = schema

        # Try to pull parameter descriptions from a NumPy-style docstring
        doc = inspect.getdoc(fn) or ""
        param_docs = _parse_param_docs(doc, list(sig.parameters.keys()))

        for p_name, p_desc in param_docs.items():
            if p_name in properties and p_desc:
                properties[p_name]["description"] = p_desc

        # Top-level function description: first paragraph of docstring
        description = doc.split("\n\n")[0].strip() if doc else f"Call the {fn.__name__} tool"

        schemas.append({
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return schemas


def _build_system_prompt() -> str:
    """Return the shared system prompt describing tools and data limits."""
    demo_hints = "; ".join(
        f"{info['description']} -> {info['path']} with bbox {info['bbox']}"
        for info in DEMO_REGIONS.values()
    )
    germany_hints = "; ".join(
        f"{key} -> {path}" for key, path in DEMO_GERMANY_MONTHLY.items()
    )
    return (
        "You are an expert climate scientist and AI agent with diagnostic, "
        "statistical, metadata and code-interpreter tools. "
        "You should be as flexible and helpful as possible: figure out which "
        "tool(s) to call, and with which arguments, purely from the user's question. "
        "The only hard limits are the spatial and temporal extent of the underlying "
        "data (described below). If a question naturally requires more than one "
        "step or figure, call the relevant tools in sequence rather than asking the "
        "user to narrow their request. "
        "For factual questions about datasets, regions, ETCCDI indices or CMIP6, "
        "call answer_climate_question first. If no existing tool covers the request, "
        "use execute_xarray_script as a fallback. "
        "After running tool(s), summarise the results in a few sentences, "
        "mentioning every figure path produced."
        "\n\nAvailable tools:\n"
        "- `compute_and_plot_extreme`: spatial extreme-index maps (RX1day, RX5day, R95p) "
        f"over an arbitrary bounding box, for a single year (2013). Datasets: {demo_hints}.\n"
        "- `regional_precip_trend`: 20-year (1995-2014) area-averaged precipitation "
        f"trend over Germany, from CPC and/or CMIP6 (dataset='cpc'/'cmip6'/'both'). "
        f"Files: {germany_hints}.\n"
        "- `germany_climatology_map`: 20-year mean precipitation over Germany. "
        "When dataset='both' it produces one 3-panel figure (CPC, CMIP6, bias). "
        "dataset='cpc'/'cmip6' gives a single mean map; metric='bias' gives a single "
        "CMIP6-minus-CPC map.\n"
        "- `global_climatology_map`: 2013 global mean precipitation. World maps use "
        "Robinson projection. When dataset='both' it produces one 3-panel figure "
        "(CPC, CMIP6, bias). dataset='cpc'/'cmip6' gives a single mean map; "
        "metric='bias' gives a single bias map.\n"
        "- `compare_precip_at_point`: precipitation comparison at a specific "
        "latitude/longitude (e.g. a city) within Germany, from CPC and/or CMIP6. "
        f"Germany's bbox is {DEMO_GERMANY_BBOX}.\n"
        "- `linear_regression_analysis`: fit a linear trend with p-value, R² and 95% CI "
        "to a precipitation time series at a point or averaged over a region.\n"
        "- `spatial_pattern_correlation`: Pearson and area-weighted pattern correlation "
        "between two gridded fields, optionally over a region.\n"
        "- `bias_metrics`: RMSE, MAE, mean error and correlation between two gridded "
        "fields; produces a bias map figure if a spatial comparison is requested.\n"
        "- `fit_precip_distribution`: fit a distribution (gamma, weibull_min, gumbel_r, "
        "norm, lognorm, expon) to a precipitation sample and return parameters, "
        "KS p-value, AIC/BIC and a PDF plot.\n"
        "- `answer_climate_question`: vector-store / RAG for factual questions about "
        "datasets, regions, ETCCDI indices (RX1day, RX5day, R95p, PRCPTOT), CMIP6 model "
        "info and data limitations (e.g. 'which dataset has daily data?', 'what is RX1day?').\n"
        "- `execute_xarray_script`: fallback code interpreter. Use ONLY when no other tool "
        "covers the request AND the answer can be computed from the demo precipitation "
        "datasets above. Write a short xarray/pandas/matplotlib script (no imports; "
        "the sandbox provides xr, np, pd, plt, scipy, DATASETS, load_dataset, save_figure). "
        "It runs in a sandbox and returns stdout + saved figures. "
        "DO NOT use this tool for climate modes, teleconnections or external indices "
        "(NAO, ENSO, AO, PDO, AMO, etc.) that are not in the demo data."
        "\nData limitations to be transparent about: only 2013 has full daily "
        "global/regional coverage (for extreme indices); only Germany has a "
        "multi-year (1995-2014) monthly record (for trends, climatology, and "
        "point comparisons). No NAO, ENSO, sea-level pressure or other "
        "teleconnection indices are available. If a question falls outside these "
        "limits, or asks about climate modes, call answer_climate_question to "
        "explain the limitation rather than running a tool or inventing results."
    )


def _execute_tool(name: str, args: dict) -> str:
    """Execute a tool by name with the given arguments."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"Error: Unknown tool '{name}'."
    try:
        return fn(**args)
    except Exception as exc:
        return f"Error executing {name}: {exc}"


def _execute_function_call(function_call) -> str:
    """Execute a Gemini FunctionCall and return its result as text."""
    name = function_call.name

    # Convert protobuf Struct to plain JSON-serializable Python types
    try:
        from google.protobuf.json_format import MessageToDict

        args = MessageToDict(function_call.args)
    except Exception:
        args = dict(function_call.args)

    return _execute_tool(name, args)


def _run_gemini_agent(user_query: str, api_key: str) -> str:
    """Run the Google Gemini function-calling agent."""
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is required for the Gemini agent. "
            "Install with: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name="gemini-flash-latest",
        tools=TOOLS,
    )
    system_prompt = _build_system_prompt()

    chat = model.start_chat()
    response = chat.send_message(
        [system_prompt, user_query],
        tool_config={
            "function_calling_config": {
                "mode": "auto",
            }
        },
    )

    for _ in range(3):
        round_results = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    result = _execute_function_call(part.function_call)
                    round_results.append(result)

        if not round_results:
            break

        response = chat.send_message(
            "Tool results:\n" + "\n".join(round_results) + "\n\n"
            "If the user\'s question requires additional tool calls, make them now. "
            "Otherwise, give your final summary."
        )
    else:
        response = chat.send_message(
            "Please stop calling tools now and give your final summary of the "
            "results produced so far, including every figure path."
        )

    try:
        return response.text
    except Exception as exc:
        return f"The Gemini agent encountered an error: {exc}"


def _run_groq_agent(user_query: str, api_key: str, model: str = "openai/gpt-oss-20b") -> str:
    """Run the Groq (OpenAI-compatible) function-calling agent.

    Groq provides an OpenAI-compatible chat completions endpoint; we use the
    `openai` package with Groq's base URL. The default model supports tool use.
    """
    try:
        from openai import OpenAI, BadRequestError
    except ImportError as exc:
        raise ImportError(
            "openai is required for the Groq agent. "
            "Install with: pip install openai"
        ) from exc

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    tools = _build_tool_schemas(TOOLS)
    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": user_query},
    ]

    try:
        for _ in range(3):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )

            message = response.choices[0].message
            if not message.tool_calls:
                return message.content or "The agent did not return a summary."

            # Build the assistant message manually because Groq's API rejects
            # extra fields (e.g. "annotations") that model_dump() may include.
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = _execute_tool(tool_call.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # Force a final text-only summary if the model keeps calling tools
        messages.append({
            "role": "user",
            "content": "Please stop calling tools now and give your final summary of the "
                       "results produced so far, including every figure path.",
        })
        final = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="none",
            temperature=0.2,
        )
        return final.choices[0].message.content or "The agent did not return a final summary."
    except BadRequestError as exc:
        # Some Groq models generate malformed tool-call JSON (e.g. long
        # multi-line scripts). Fall back to a plain text answer instead.
        if "tool_use_failed" in str(exc) or "Failed to parse" in str(exc):
            try:
                fallback = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _build_system_prompt()},
                        {"role": "user", "content": user_query},
                        {
                            "role": "user",
                            "content": (
                                "You tried to call a tool but the tool-call "
                                "arguments were not valid. Do not call any "
                                "tools. Answer the question directly, "
                                "explaining any data limitations if the "
                                "request is outside the available datasets."
                            ),
                        },
                    ],
                    temperature=0.2,
                )
                return fallback.choices[0].message.content or "The agent did not return a summary."
            except Exception as inner:
                return f"The Groq agent encountered a tool-calling error and the fallback also failed: {inner}"
        return f"The Groq agent encountered an error: {exc}"
    except Exception as exc:
        return f"The Groq agent encountered an error: {exc}"


def run_climate_agent(
    user_query: str,
    api_key: str | None = None,
    provider: str = "gemini",
    groq_model: str = "openai/gpt-oss-20b",
) -> str:
    """Run the climate ReAct agent against the user query.

    Parameters
    ----------
    user_query
        Natural-language scientific question.
    api_key
        Optional API key. If omitted, reads from {PROVIDER}_API_KEY env var
        or Streamlit secrets.
    provider
        LLM provider to use: "gemini" or "groq".
    groq_model
        Model name to use when provider="groq".

    Returns
    -------
    str
        Final agent response, including the path(s) to any generated figure(s).
    """
    provider = provider.lower().strip()
    if provider not in ("gemini", "groq"):
        return f"Unsupported provider '{provider}'. Choose 'gemini' or 'groq'."

    if api_key is None:
        api_key = _get_api_key(provider)
    if not api_key:
        return (
            f"No {provider.upper()} API key found. Set {provider.upper()}_API_KEY "
            "in your environment or add it to .streamlit/secrets.toml to run the agent."
        )

    if provider == "gemini":
        return _run_gemini_agent(user_query, api_key)
    return _run_groq_agent(user_query, api_key, model=groq_model)


class ClimateDiagnosticAgent:
    """Backwards-compatible wrapper around run_climate_agent."""

    def __init__(self, data_dir: Path, figures_dir: Path):
        self.data_dir = data_dir
        self.figures_dir = figures_dir

    def plan(self, question: str, provider: str = "gemini") -> list[dict]:
        """Return a single-step plan for the agent."""
        return [
            {
                "tool": "compute_and_plot_extreme",
                "args": {"user_query": question},
                "reason": f"Execute the user question via the {provider} ReAct agent",
            }
        ]

    def execute_plan(self, plan: list[dict], provider: str = "gemini") -> dict:
        """Execute the plan by calling the selected agent."""
        user_query = plan[0].get("args", {}).get("user_query", "")
        result = run_climate_agent(user_query, provider=provider)
        return {"results": [{"step": plan[0], "result": result}]}
