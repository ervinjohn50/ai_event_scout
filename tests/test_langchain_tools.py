"""Tests that LangChain tools behave exactly like the registry they wrap."""

import json
from datetime import date

import pytest

pytest.importorskip("langchain_core", reason="needs the langchain extra")

from langchain_anthropic.chat_models import convert_to_anthropic_tool  # noqa: E402

from event_scout.events import SampleEventSource  # noqa: E402
from event_scout.langchain_tools import to_langchain_tools  # noqa: E402
from event_scout.shortlist import ShortlistStore  # noqa: E402
from event_scout.tools import build_tools  # noqa: E402

TODAY = date(2026, 11, 3)


@pytest.fixture
def registry(tmp_path):
    return build_tools(SampleEventSource(today=TODAY), ShortlistStore(tmp_path / "shortlist.json"))


@pytest.fixture
def tools(registry):
    return {tool.name: tool for tool in to_langchain_tools(registry)}


def call(tool, args):
    """Invoke a tool the way an agent does, with a tool call, and get its ToolMessage."""
    return tool.invoke({"type": "tool_call", "name": tool.name, "args": args, "id": "toolu_1"})


def test_claude_sees_identical_tool_definitions_from_both_engines(registry):
    converted = [convert_to_anthropic_tool(tool) for tool in to_langchain_tools(registry)]
    assert converted == registry.definitions()


def test_valid_call_returns_the_same_json_as_the_registry(registry, tools):
    args = {"city": "Chicago", "category": "sports"}
    message = call(tools["search_events"], args)

    assert message.status == "success"
    assert message.tool_call_id == "toolu_1"
    assert message.content == registry.run("search_events", args).content
    assert json.loads(message.content)["events"][0]["id"] == "sample-003"


def test_invalid_input_returns_the_same_error_as_the_registry(registry, tools):
    args = {"town": "Chicago", "max_results": 500}
    message = call(tools["search_events"], args)

    assert message.status == "error"
    assert message.content == registry.run("search_events", args).content
    assert "max_results" in message.content and "town" in message.content


def test_handler_error_comes_back_as_an_error_message(tools):
    message = call(tools["get_event_details"], {"event_id": "sample-999"})
    assert message.status == "error"
    assert "sample-999" in message.content


def test_tool_with_no_input(tools):
    message = call(tools["list_shortlist"], {})
    assert message.status == "success"
    assert json.loads(message.content) == {"count": 0, "entries": []}


def test_tools_share_state_with_the_registry(registry, tools):
    message = call(tools["save_to_shortlist"], {"event_id": "sample-002", "note": "date night"})
    assert message.status == "success"
    assert json.loads(registry.run("list_shortlist", {}).content)["count"] == 1
