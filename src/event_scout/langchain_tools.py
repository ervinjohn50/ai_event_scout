"""Expose the existing tools to LangChain.

Each LangChain tool is a thin wrapper around `ToolRegistry`, so both engines share the
same handlers, validation, and error messages.

The tool's `args_schema` is the JSON Schema from `ToolRegistry.definitions()`, not the
Pydantic model. Given a plain schema, LangChain sends it to Claude unchanged and skips
its own validation. That means Claude sees identical tool definitions from both
engines, and `ToolRegistry.run` stays the only place input is validated.

LangGraph runs the tool calls from one model reply on separate threads, while the
hand-written loop runs them one at a time. The handlers aren't written for concurrency:
two threads saving to the shortlist at once each read the file, add an entry, and write
it back, so one overwrites the other. A lock makes both engines run tools one at a time.
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.tools import StructuredTool, ToolException

from .tools import ToolRegistry


def to_langchain_tools(registry: ToolRegistry) -> list[StructuredTool]:
    lock = threading.Lock()
    return [_wrap(registry, definition, lock) for definition in registry.definitions()]


def _wrap(registry: ToolRegistry, definition: dict[str, Any], lock: threading.Lock) -> StructuredTool:
    name = definition["name"]

    def run(**tool_input: Any) -> str:
        with lock:
            outcome = registry.run(name, tool_input)
        if outcome.is_error:
            # LangChain's way of marking a failed tool call. With handle_tool_error=True
            # the message goes back to the model instead of raising out of the agent,
            # which is the same as is_error=True on a tool_result block.
            raise ToolException(outcome.content)
        return outcome.content

    return StructuredTool.from_function(
        func=run,
        name=name,
        description=definition["description"],
        args_schema=definition["input_schema"],
        handle_tool_error=True,
    )
