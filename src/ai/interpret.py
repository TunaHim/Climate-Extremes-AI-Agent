"""AI figure interpretation with Gemini, Groq and Ollama."""

import base64
import json
from pathlib import Path

import requests


def _encode_image(image_path: Path) -> str:
    """Encode an image to base64."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _load_metadata(image_path: Path) -> dict:
    """Load JSON sidecar metadata if present."""
    sidecar = image_path.with_suffix(".json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text())
        except Exception:
            pass
    return {}


def ask_gemini(image_path: Path, prompt: str, api_key: str | None = None) -> str:
    """Ask Gemini 2.5 Flash to interpret a figure."""
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError("google-generativeai is required for Gemini. Install with: pip install google-generativeai") from exc

    if api_key is None:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY", None)
        if not api_key:
            return "❌ No Gemini API key found. Add GEMINI_API_KEY to Streamlit secrets."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    image_part = {"mime_type": "image/png", "data": image_path.read_bytes()}
    metadata = _load_metadata(image_path)
    context = ""
    if metadata:
        context = "\n\nFigure metadata:\n" + json.dumps(metadata, indent=2)
    response = model.generate_content([image_part, prompt + context])
    return response.text


def ask_ollama(image_path: Path, prompt: str, model_name: str = "ministral3", local: bool = True) -> str:
    """Ask a local Ollama vision model to interpret a figure."""
    url = "http://localhost:11434/api/chat" if local else "https://api.ollama.com/api/chat"
    api_key = None
    if not local:
        try:
            import streamlit as st
            api_key = st.secrets.get("OLLAMA_API_KEY", None)
        except Exception:
            pass

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    metadata = _load_metadata(image_path)
    context = ""
    if metadata:
        context = "\n\nFigure metadata:\n" + json.dumps(metadata, indent=2)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt + context,
                "images": [_encode_image(image_path)],
            }
        ],
        "stream": False,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", result.get("response", "No response"))
    except requests.exceptions.ConnectionError:
        return "❌ Could not connect to Ollama. Make sure `ollama serve` is running."
    except requests.exceptions.Timeout:
        return "⏳ Ollama request timed out. Try a smaller figure or a faster model."
    except Exception as exc:
        return f"❌ Error calling Ollama: {exc}"


def ask_groq(image_path: Path, prompt: str, model_name: str = "llama-3.2-11b-vision-preview") -> str:
    """Ask a Groq vision model to interpret a figure.

    Uses the OpenAI-compatible chat completions endpoint on Groq.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai is required for Groq interpretation. Install with: pip install openai"
        ) from exc

    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        import os
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "❌ No GROQ_API_KEY found. Add it to Streamlit secrets or your environment."

    metadata = _load_metadata(image_path)
    context = ""
    if metadata:
        context = "\n\nFigure metadata:\n" + json.dumps(metadata, indent=2)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    encoded = _encode_image(image_path)
    data_url = f"data:image/png;base64,{encoded}"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + context},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.4,
            max_tokens=1024,
        )
        return response.choices[0].message.content or "The model returned an empty response."
    except Exception as exc:
        return f"❌ Error calling Groq: {exc}"


def interpret_figure(
    image_path: Path,
    prompt: str,
    model: str = "Gemini 2.5 Flash",
) -> str:
    """Dispatch interpretation to the selected model."""
    if "gemini" in model.lower():
        return ask_gemini(image_path, prompt)
    if "groq" in model.lower():
        return ask_groq(image_path, prompt)
    if "ministral" in model.lower():
        return ask_ollama(image_path, prompt, model_name="ministral3")
    if "gemma" in model.lower():
        return ask_ollama(image_path, prompt, model_name="gemma4")
    if "qwen" in model.lower():
        return ask_ollama(image_path, prompt, model_name="qwen2.5vl:7b")
    return f"❌ Unknown model: {model}"
