"""Optional LangSmith tracing: records every question as a trace in a web dashboard.

Tracing is off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are both set. When it's
on, prompts, tool results, and answers are sent to LangSmith's servers.

Each engine gets traced differently, without changing either one:
- LangChain and LangGraph report to LangSmith on their own.
- The hand-written engine gets a wrapped API client, which traces each model call, and
  a wrapped tool registry, which traces each tool call.
`TracedEngine` then wraps either engine, so all the steps of one question nest under a
single trace tagged with the engine's name.

Tracing must never affect the app: a LangSmith failure is logged, and the answer still
comes back. `langsmith` is imported inside functions because it's an optional package.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

from .agent import AgentReply
from .tools import ToolOutcome, ToolRegistry

TRACE_NAME = "event-scout"
DEFAULT_PROJECT = "event-scout"


def tracing_requested(env: Mapping[str, str]) -> bool:
    return env.get("LANGSMITH_TRACING", "").strip().lower() == "true"


def langsmith_enabled(env: Mapping[str, str]) -> bool:
    return tracing_requested(env) and bool(env.get("LANGSMITH_API_KEY"))


def configure_tracing(env: MutableMapping[str, str] = os.environ) -> bool:
    """Decide whether tracing is on. A setup problem prints a warning, but the app still runs."""
    if not tracing_requested(env):
        return False
    if not langsmith_enabled(env):
        # LangChain only checks LANGSMITH_TRACING, so it would try to upload without a key.
        env["LANGSMITH_TRACING"] = "false"
        print("Warning: LANGSMITH_TRACING is on but LANGSMITH_API_KEY isn't set. Tracing is off.", file=sys.stderr)
        return False
    try:
        import langsmith  # noqa: F401
    except ImportError:
        env["LANGSMITH_TRACING"] = "false"
        print("Warning: tracing needs the langsmith package (uv sync --extra langchain). Tracing is off.", file=sys.stderr)
        return False
    env.setdefault("LANGSMITH_PROJECT", DEFAULT_PROJECT)
    return True


def trace_client(client: Any) -> Any:
    """Make the Anthropic client report each model call. Requests and replies are unchanged."""
    from langsmith.wrappers import wrap_anthropic

    return wrap_anthropic(client)


class TracedToolRegistry:
    """Reports each tool call, then runs it with the registry it wraps."""

    def __init__(self, tools: ToolRegistry) -> None:
        self._tools = tools

    def definitions(self) -> list[dict[str, Any]]:
        return self._tools.definitions()

    def run(self, name: str, raw_input: dict[str, Any]) -> ToolOutcome:
        from langsmith import traceable

        def run_tool(tool_input: dict[str, Any]) -> ToolOutcome:
            return self._tools.run(name, tool_input)

        return traceable(name=name, run_type="tool", process_outputs=_outcome_as_dict)(run_tool)(raw_input)


class TracedEngine:
    """Wraps an engine so that each question becomes one trace. Same interface as the engine."""

    def __init__(self, agent: Any, engine_name: str) -> None:
        from langsmith import traceable

        self.agent = agent
        self.config = agent.config
        self._traced_ask = traceable(
            name=TRACE_NAME,
            run_type="chain",
            tags=[f"engine:{engine_name}"],
            metadata={"engine": engine_name, "model": agent.config.model, "effort": agent.config.effort},
            process_outputs=_reply_as_dict,
        )(agent.ask)

    def ask(self, user_text: str) -> AgentReply:
        return self._traced_ask(user_text)

    def reset(self) -> None:
        self.agent.reset()

    def export_messages(self) -> list[dict[str, Any]]:
        return self.agent.export_messages()


def flush_traces() -> None:
    """Wait for queued traces to upload.

    Uploads happen on background threads, so a short run could exit before they finish.
    A failure here is ignored, because a tracing problem shouldn't affect the app.
    """
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        from langsmith.run_trees import get_cached_client

        wait_for_all_tracers()
        get_cached_client().flush()
    except Exception:
        pass


def _reply_as_dict(reply: AgentReply) -> dict[str, Any]:
    return {"answer": reply.text, "stop_reason": reply.stop_reason, "steps": reply.steps}


def _outcome_as_dict(outcome: ToolOutcome) -> dict[str, Any]:
    return {"content": outcome.content, "is_error": outcome.is_error}
