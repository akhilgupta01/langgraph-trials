"""Helpers to configure Gemini API vs Vertex AI model providers.

This module centralizes provider selection so backend agents can use one of:
- Gemini Developer API
- Vertex AI Gemini

Provider selection is driven by `GOOGLE_LLM_PROVIDER`:
- `gemini` -> force Gemini Developer API
- `vertex` -> force Vertex AI
- `auto`   -> prefer Vertex when GCP project configuration is present,
              otherwise fallback to Gemini Developer API
"""

import os
from typing import Any, Literal

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

Provider = Literal["gemini", "vertex"]


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_provider() -> Provider:
    configured = os.getenv("GOOGLE_LLM_PROVIDER", "auto").strip().lower()

    if configured in {"gemini", "google", "google_genai", "gemini_api"}:
        return "gemini"
    if configured in {"vertex", "vertex_ai"}:
        return "vertex"
    if configured == "auto":
        # Vertex is selected automatically when project configuration is present.
        if _is_truthy(os.getenv("GOOGLE_GENAI_USE_VERTEXAI")):
            return "vertex"
        if os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_PROJECT_ID"):
            return "vertex"
        return "gemini"

    raise ValueError(
        "Unsupported GOOGLE_LLM_PROVIDER. Use one of: auto, gemini, vertex."
    )


def _vertex_project() -> str | None:
    return os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID")


def _vertex_location() -> str | None:
    return os.getenv("VERTEX_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GCP_LOCATION")


def default_model_name() -> str:
    return os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash")


def build_chat_model(
    *,
    model_name: str,
    temperature: float | None = None,
    cached_content: str | None = None,
) -> ChatGoogleGenerativeAI:
    provider = _resolve_provider()

    kwargs: dict[str, Any] = {"model": model_name}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if cached_content is not None:
        kwargs["cached_content"] = cached_content

    if provider == "vertex":
        kwargs["vertexai"] = True
        project = _vertex_project()
        location = _vertex_location()
        if project:
            kwargs["project"] = project
        if location:
            kwargs["location"] = location
    else:
        # Explicitly disable Vertex auto-detection when Gemini is selected.
        kwargs["vertexai"] = False

    return ChatGoogleGenerativeAI(**kwargs)


def create_genai_client() -> genai.Client:
    provider = _resolve_provider()

    if provider == "vertex":
        kwargs: dict[str, Any] = {"vertexai": True}
        project = _vertex_project()
        location = _vertex_location()
        if project:
            kwargs["project"] = project
        if location:
            kwargs["location"] = location
        return genai.Client(**kwargs)

    return genai.Client(vertexai=False)
