"""Tests for LangSmith tracing. No account or network is used.

A recording client stands in for LangSmith's client and keeps whatever would have been
uploaded. Custom engine tests use the real Anthropic SDK with a fake network layer,
because `wrap_anthropic` patches the real client.
"""

import json
import os
import socket
from datetime import date
from unittest.mock import MagicMock

import pytest

pytest.importorskip("langsmith", reason="needs the langchain extra")

import anthropic  # noqa: E402
import httpx2  # noqa: E402
from fakes import FakeClient, response, text, thinking, tool_use  # noqa: E402
from fakes_langchain import scripted_model  # noqa: E402
from langsmith import Client  # noqa: E402
from langsmith.run_helpers import tracing_context  # noqa: E402

from event_scout import cli, tracing  # noqa: E402
from event_scout.agent import ScoutAgent  # noqa: E402
from event_scout.config import AgentConfig  # noqa: E402
from event_scout.events import SampleEventSource  # noqa: E402
from event_scout.langgraph_agent import LangGraphScoutAgent  # noqa: E402
from event_scout.shortlist import ShortlistStore  # noqa: E402
from event_scout.tools import build_tools  # noqa: E402

TODAY = date(2026, 11, 3)
SYSTEM = "You are a test scout."


@pytest.fixture
def registry(tmp_path):
    return build_tools(SampleEventSource(today=TODAY), ShortlistStore(tmp_path / "shortlist.json"))


def script():
    return [
        response(thinking("Search."), tool_use("t1", "search_events", {"city": "Austin"}), stop_reason="tool_use"),
        response(text("Austin has shows.")),
    ]


def fake_network_client(responses, bodies=None):
    """A real Anthropic client whose HTTP layer replays scripted responses."""
    remaining = list(responses)

    def handler(request):
        if bodies is not None:
            bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=remaining.pop(0).model_dump(mode="json", exclude_none=True))

    return anthropic.Anthropic(api_key="fake-key", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


class Recorder:
    """Stands in for LangSmith's client, and keeps each run instead of uploading it."""

    def __init__(self):
        self.runs = {}
        self.client = MagicMock(spec=Client)
        self.client.tracing_sampling_rate = None
        self.client.create_run.side_effect = self._create
        self.client.update_run.side_effect = self._update

    def _create(self, *args, **kwargs):
        name = args[0] if args else kwargs.get("name")
        run_type = args[2] if len(args) > 2 else kwargs.get("run_type")
        self.runs[str(kwargs["id"])] = {**kwargs, "name": name, "run_type": run_type}

    def _update(self, *args, **kwargs):
        run_id = args[0] if args else kwargs.pop("run_id")
        # An update sends inputs=None to mean "unchanged", so None mustn't overwrite values.
        self.runs[str(run_id)].update({k: v for k, v in kwargs.items() if v is not None})

    def by_type(self, run_type):
        return [r for r in self.runs.values() if r["run_type"] == run_type]

    def root(self):
        [root] = [r for r in self.runs.values() if not r.get("parent_run_id")]
        return root

    def in_order(self, parent):
        children = [r for r in self.runs.values() if r.get("parent_run_id") and str(r["parent_run_id"]) == str(parent["id"])]
        return sorted(children, key=lambda r: r["dotted_order"])


def traced_custom(responses, registry, bodies=None):
    client = tracing.trace_client(fake_network_client(responses, bodies))
    agent = ScoutAgent(client, tracing.TracedToolRegistry(registry), SYSTEM)
    return tracing.TracedEngine(agent, "custom")


def traced_langgraph(responses, registry):
    agent = LangGraphScoutAgent(scripted_model(responses), registry, SYSTEM)
    return tracing.TracedEngine(agent, "langgraph")


# --- Turning tracing on and off ------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"LANGSMITH_API_KEY": "k"}, False),
        ({"LANGSMITH_TRACING": "true"}, False),
        ({"LANGSMITH_TRACING": "false", "LANGSMITH_API_KEY": "k"}, False),
        ({"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": ""}, False),
        ({"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "k"}, True),
        ({"LANGSMITH_TRACING": "TRUE", "LANGSMITH_API_KEY": "k"}, True),
    ],
)
def test_langsmith_enabled(env, expected):
    assert tracing.langsmith_enabled(env) is expected


def test_configure_is_silent_when_tracing_was_not_requested(capsys):
    assert tracing.configure_tracing({}) is False
    assert capsys.readouterr().err == ""


def test_configure_turns_tracing_off_and_warns_without_a_key(capsys):
    env = {"LANGSMITH_TRACING": "true"}
    assert tracing.configure_tracing(env) is False
    # LangChain reads this variable itself, so it has to be switched off explicitly.
    assert env["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY" in capsys.readouterr().err


def test_configure_sets_a_default_project_but_keeps_yours():
    env = {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "k"}
    assert tracing.configure_tracing(env) is True
    assert env["LANGSMITH_PROJECT"] == "event-scout"

    env = {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "k", "LANGSMITH_PROJECT": "mine"}
    tracing.configure_tracing(env)
    assert env["LANGSMITH_PROJECT"] == "mine"


def test_configure_does_not_add_a_project_when_an_agent_is_set():
    """LangSmith doesn't trace calls when an agent and a project are both set."""
    env = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "k",
        "LANGSMITH_AGENT_ID": "event-scout",
        "LANGSMITH_AGENT_ENVIRONMENT": "local",
    }
    assert tracing.configure_tracing(env) is True
    assert "LANGSMITH_PROJECT" not in env


def test_configure_warns_and_continues_if_langsmith_is_not_installed(monkeypatch, capsys):
    monkeypatch.setitem(__import__("sys").modules, "langsmith", None)
    env = {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "k"}
    assert tracing.configure_tracing(env) is False
    assert env["LANGSMITH_TRACING"] == "false"
    assert "uv sync --extra langchain" in capsys.readouterr().err


# --- What a trace contains ------------------------------------------------------------------


def test_custom_engine_produces_one_nested_trace(registry):
    recorder = Recorder()
    with tracing_context(enabled=True, client=recorder.client):
        reply = traced_custom(script(), registry).ask("What's in Austin?")

    assert reply.text == "Austin has shows."
    root = recorder.root()
    assert root["name"] == "event-scout"
    assert root["tags"] == ["engine:custom"]
    assert [(r["run_type"], r["name"]) for r in recorder.in_order(root)] == [
        ("llm", "ChatAnthropic"),
        ("tool", "search_events"),
        ("llm", "ChatAnthropic"),
    ]


def test_langgraph_engine_produces_one_nested_trace(registry):
    recorder = Recorder()
    with tracing_context(enabled=True, client=recorder.client):
        reply = traced_langgraph(script(), registry).ask("What's in Austin?")

    assert reply.text == "Austin has shows."
    root = recorder.root()
    assert root["name"] == "event-scout"
    assert root["tags"] == ["engine:langgraph"]
    assert len(recorder.by_type("llm")) == 2
    assert [r["name"] for r in recorder.by_type("tool")] == ["search_events"]


def test_trace_records_the_question_answer_and_settings(registry):
    recorder = Recorder()
    with tracing_context(enabled=True, client=recorder.client):
        traced_custom(script(), registry).ask("What's in Austin?")

    root = recorder.root()
    assert root["inputs"] == {"user_text": "What's in Austin?"}
    assert root["outputs"] == {"answer": "Austin has shows.", "stop_reason": "end_turn", "steps": 2}
    metadata = root["extra"]["metadata"]
    assert (metadata["engine"], metadata["model"], metadata["effort"]) == ("custom", "claude-sonnet-5-5", "medium")


def test_tool_runs_record_input_and_errors(registry):
    recorder = Recorder()
    responses = [
        response(tool_use("t1", "get_event_details", {"event_id": "sample-999"}), stop_reason="tool_use"),
        response(text("Not found.")),
    ]
    with tracing_context(enabled=True, client=recorder.client):
        traced_custom(responses, registry).ask("details?")

    [tool_run] = recorder.by_type("tool")
    assert tool_run["inputs"] == {"tool_input": {"event_id": "sample-999"}}
    assert tool_run["outputs"]["is_error"] is True
    assert "sample-999" in tool_run["outputs"]["content"]


# --- Tracing must not affect the app --------------------------------------------------------


def test_tracing_does_not_change_requests_or_replies(registry):
    plain_bodies, traced_bodies = [], []
    plain = ScoutAgent(fake_network_client(script(), plain_bodies), registry, SYSTEM)
    traced = traced_custom(script(), registry, traced_bodies)

    plain_reply = plain.ask("What's in Austin?")
    with tracing_context(enabled=True, client=Recorder().client):
        traced_reply = traced.ask("What's in Austin?")

    assert traced_bodies == plain_bodies  # byte-for-byte, including thinking blocks
    assert traced_reply == plain_reply
    assert traced.export_messages() == plain.export_messages()


def test_traced_engine_passes_through_reset_and_config(registry):
    traced = traced_langgraph([response(text("a")), response(text("b"))], registry)
    assert traced.config.model == "claude-sonnet-5-5"
    with tracing_context(enabled=True, client=Recorder().client):
        traced.ask("first")
        assert len(traced.export_messages()) == 2
        traced.reset()
    assert traced.export_messages() == []


@pytest.mark.parametrize("make_engine", [traced_custom, traced_langgraph])
def test_a_langsmith_failure_does_not_break_the_agent(make_engine, registry):
    broken = MagicMock(spec=Client)
    broken.tracing_sampling_rate = None
    broken.create_run.side_effect = RuntimeError("LangSmith is down")
    broken.update_run.side_effect = RuntimeError("LangSmith is down")

    with tracing_context(enabled=True, client=broken):
        reply = make_engine(script(), registry).ask("What's in Austin?")

    assert reply.text == "Austin has shows."


def test_a_flush_failure_is_ignored(monkeypatch):
    monkeypatch.setattr("langsmith.run_trees.get_cached_client", MagicMock(side_effect=RuntimeError("down")))
    tracing.flush_traces()


# --- The CLI -----------------------------------------------------------------------------------


@pytest.fixture
def isolated_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "TICKETMASTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("builtins.input", MagicMock(side_effect=EOFError))


@pytest.fixture
def flush_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "flush_traces", lambda: calls.append(1))
    return calls


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test on any attempt to open a network connection."""

    def refuse(self, address):
        raise AssertionError(f"Unexpected network connection to {address}")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.mark.parametrize("engine", ["custom", "langgraph"])
def test_with_tracing_off_nothing_is_traced_or_sent(engine, isolated_cli, flush_calls, no_network, capsys):
    kwargs = {"client": FakeClient(script())} if engine == "custom" else {"chat_model": scripted_model(script())}

    assert cli.main(["--engine", engine, "--quiet", "What's in Austin?"], **kwargs) == 0

    assert "Austin has shows." in capsys.readouterr().out
    assert flush_calls == []


@pytest.mark.parametrize("engine", ["custom", "langgraph"])
def test_cli_traces_both_engines_when_tracing_is_on(engine, isolated_cli, monkeypatch, flush_calls):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")
    kwargs = {"client": fake_network_client(script())} if engine == "custom" else {"chat_model": scripted_model(script())}
    recorder = Recorder()

    with tracing_context(enabled=True, client=recorder.client):
        assert cli.main(["--engine", engine, "--quiet", "What's in Austin?"], **kwargs) == 0

    assert recorder.root()["tags"] == [f"engine:{engine}"]
    assert flush_calls == [1]  # queued traces are uploaded before exit


def test_banner_shows_tracing_status(isolated_cli, monkeypatch, flush_calls, capsys):
    cli.main(["--quiet"], client=FakeClient([]))
    assert "LangSmith tracing: off." in capsys.readouterr().out

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")
    cli.main(["--quiet"], client=fake_network_client([]))
    assert "LangSmith tracing: on." in capsys.readouterr().out
    assert flush_calls == [1]


def test_build_agent_returns_plain_engines_when_tracing_is_off(registry):
    agent = cli.build_agent("custom", registry, SYSTEM, AgentConfig(), None, client=FakeClient([]))
    assert isinstance(agent, ScoutAgent)


def test_test_setup_clears_langsmith_settings_from_the_shell():
    assert "LANGSMITH_TRACING" not in os.environ
    assert "LANGSMITH_API_KEY" not in os.environ
