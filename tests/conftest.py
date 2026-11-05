"""Settings shared by every test."""

import pytest


@pytest.fixture(autouse=True)
def no_real_tracing(monkeypatch):
    """Keep LangSmith settings in your shell from uploading test runs to your account."""
    for key in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_AGENT_ID",
        "LANGSMITH_AGENT_ENVIRONMENT",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    try:
        from langsmith.utils import get_env_var
    except ImportError:
        return
    get_env_var.cache_clear()  # LangSmith caches environment lookups
