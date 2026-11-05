"""Tests for the LangGraph engine.

Many tests here run one script through BOTH engines and compare the results, because
the point of the engine is to behave like the hand-written loop. The LangGraph side runs
the real ChatAnthropic and create_agent code, with only the network call replaced.
"""

import json
from datetime import date

import pytest

pytest.importorskip("langchain", reason="needs the langchain extra")

from fakes import FakeClient, response, text, thinking, tool_use  # noqa: E402
from fakes_langchain import scripted_model  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from event_scout import cli  # noqa: E402
from event_scout.agent import AgentObserver, ScoutAgent  # noqa: E402
from event_scout.config import FALLBACK_BETA, AgentConfig  # noqa: E402
from event_scout.events import SampleEventSource  # noqa: E402
from event_scout.langgraph_agent import LangGraphScoutAgent, _usage  # noqa: E402
from event_scout.shortlist import ShortlistStore  # noqa: E402
from event_scout.tools import build_tools  # noqa: E402

TODAY = date(2026, 11, 3)
SYSTEM = "You are a test scout."


@pytest.fixture
def registry(tmp_path):
    return build_tools(SampleEventSource(today=TODAY), ShortlistStore(tmp_path / "shortlist.json"))


def make_langgraph(responses, registry, **config):
    agent_config = AgentConfig(**config)
    model = scripted_model(responses, agent_config)
    return LangGraphScoutAgent(model, registry, SYSTEM, agent_config), model


def make_handwritten(responses, registry, **config):
    client = FakeClient(responses)
    return ScoutAgent(client, registry, SYSTEM, AgentConfig(**config)), client


# --- Both engines, same script, same results --------------------------------------------------

SCRIPTS = {
    "direct answer": lambda: [response(thinking("Easy."), text("Hello!"))],
    "one tool, then answer": lambda: [
        response(thinking("Search."), tool_use("t1", "search_events", {"city": "Austin"}), stop_reason="tool_use"),
        response(text("Austin has shows.")),
    ],
    "two tools in one reply": lambda: [
        response(
            tool_use("t1", "search_events", {"city": "Austin"}),
            tool_use("t2", "get_event_details", {"event_id": "sample-001"}),
            stop_reason="tool_use",
        ),
        response(text("Found them.")),
    ],
    "tool errors, then recovery": lambda: [
        response(tool_use("t1", "search_events", {"max_results": 500}), stop_reason="tool_use"),
        response(tool_use("t2", "search_events", {"max_results": 5}), stop_reason="tool_use"),
        response(text("Fixed it.")),
    ],
    "refusal": lambda: [response(stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"})],
    "cut off with text only": lambda: [response(text("Partial answer"), stop_reason="max_tokens")],
    "finished with no text": lambda: [response(thinking("Nothing to add."))],
    "text before a tool call": lambda: [
        response(text("Let me check."), tool_use("t1", "list_shortlist", {}), stop_reason="tool_use"),
        response(text("It's empty.")),
    ],
}


@pytest.mark.parametrize("name", SCRIPTS)
def test_both_engines_give_the_same_reply(name, registry):
    handwritten, _ = make_handwritten(SCRIPTS[name](), registry)
    langgraph, _ = make_langgraph(SCRIPTS[name](), registry)

    expected = handwritten.ask("question")
    actual = langgraph.ask("question")

    assert actual.text == expected.text
    assert actual.stop_reason == expected.stop_reason
    assert actual.steps == expected.steps
    assert actual.usage == expected.usage


@pytest.mark.parametrize("name", ["one tool, then answer", "two tools in one reply", "tool errors, then recovery"])
def test_both_engines_send_the_same_tool_results_to_claude(name, registry):
    handwritten, client = make_handwritten(SCRIPTS[name](), registry)
    langgraph, model = make_langgraph(SCRIPTS[name](), registry)
    handwritten.ask("question")
    langgraph.ask("question")

    def tool_results(messages):
        blocks = [b for m in messages if isinstance(m["content"], list) for b in m["content"]]
        results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
        return sorted((r["tool_use_id"], r["content"], r["is_error"]) for r in results)

    assert tool_results(model.requests[-1]["messages"]) == tool_results(client.requests[-1]["messages"])


def test_step_limit_allows_the_same_number_of_model_calls(registry):
    def endless():
        return [response(tool_use(f"t{i}", "list_shortlist", {}), stop_reason="tool_use") for i in range(10)]

    handwritten, client = make_handwritten(endless(), registry, max_steps=3)
    langgraph, model = make_langgraph(endless(), registry, max_steps=3)

    expected = handwritten.ask("loop forever")
    actual = langgraph.ask("loop forever")

    assert len(model.requests) == len(client.requests) == 3
    assert actual.text == expected.text
    assert actual.stop_reason == expected.stop_reason == "step_limit"
    assert actual.usage == expected.usage


def test_observers_see_the_same_steps_from_both_engines(registry):
    class Recorder(AgentObserver):
        def __init__(self):
            self.events = []

        def on_model_call(self, step, messages):
            self.events.append(("call", step))

        def on_model_response(self, step, resp):
            self.events.append(("response", step, resp.stop_reason))

        def on_tool_result(self, block, outcome):
            self.events.append(("tool", block.name, outcome.is_error))

    def script():
        return SCRIPTS["two tools in one reply"]()

    first, second = Recorder(), Recorder()
    ScoutAgent(FakeClient(script()), registry, SYSTEM, observer=first).ask("q")
    LangGraphScoutAgent(scripted_model(script()), registry, SYSTEM, observer=second).ask("q")

    # Tool results can arrive in any order from LangGraph, so compare them as a set.
    def normalized(events):
        return [e for e in events if e[0] != "tool"], sorted(e for e in events if e[0] == "tool")

    assert normalized(second.events) == normalized(first.events)


def test_observers_get_the_same_message_count_with_parallel_tool_calls(registry):
    """LangChain stores one message per tool result, but the API gets them merged into one."""
    counts = {"handwritten": [], "langgraph": []}

    def recorder(name):
        class Recorder(AgentObserver):
            def on_model_call(self, step, messages):
                counts[name].append(len(messages))

        return Recorder()

    script = SCRIPTS["two tools in one reply"]
    ScoutAgent(FakeClient(script()), registry, SYSTEM, observer=recorder("handwritten")).ask("q")
    LangGraphScoutAgent(scripted_model(script()), registry, SYSTEM, observer=recorder("langgraph")).ask("q")

    assert counts["langgraph"] == counts["handwritten"] == [1, 3]


def test_trace_prints_the_same_labels_for_both_engines(registry, capsys):
    from event_scout.trace import ConsoleTrace

    script = lambda: [  # noqa: E731
        response(thinking("Search Austin."), tool_use("t1", "search_events", {"city": "Austin"}), stop_reason="tool_use"),
        response(text("Done.")),
    ]
    trace = ConsoleTrace(color=False)
    ScoutAgent(FakeClient(script()), registry, SYSTEM, observer=trace).ask("q")
    expected = capsys.readouterr().out

    LangGraphScoutAgent(scripted_model(script()), registry, SYSTEM, observer=trace).ask("q")
    assert capsys.readouterr().out == expected


# --- What's sent to the API -------------------------------------------------------------------


def test_request_matches_the_handwritten_engine(registry):
    handwritten, client = make_handwritten([response(text("Hi"))], registry, effort="low")
    langgraph, model = make_langgraph([response(text("Hi"))], registry, effort="low")
    handwritten.ask("hello")
    langgraph.ask("hello")

    actual, expected = model.requests[0], client.requests[0]
    keys = ["model", "max_tokens", "system", "tools", "thinking", "output_config", "cache_control", "betas", "fallbacks"]
    assert {k: actual.get(k) for k in keys} == {k: expected.get(k) for k in keys}
    assert actual["betas"] == [FALLBACK_BETA]
    assert actual["messages"] == [{"role": "user", "content": "hello"}]
    assert not actual.get("tool_choice")  # forced tool use is rejected on Opus 5.5
    assert "temperature" not in actual  # also rejected


def test_fallbacks_can_be_turned_off(registry):
    agent, model = make_langgraph([response(text("Hi"))], registry, use_fallbacks=False)
    agent.ask("hello")
    assert "fallbacks" not in model.requests[0]
    assert not model.requests[0].get("betas")


def test_thinking_blocks_go_back_to_claude_unchanged(registry):
    agent, model = make_langgraph(SCRIPTS["one tool, then answer"](), registry)
    agent.ask("question")

    assistant = model.requests[1]["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "thinking", "thinking": "Search.", "signature": "fake-signature"}


def test_all_tool_results_share_one_user_message(registry):
    agent, model = make_langgraph(SCRIPTS["two tools in one reply"](), registry)
    agent.ask("question")

    sent = model.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert len(sent[2]["content"]) == 2


def test_history_is_append_only_across_questions(registry):
    agent, model = make_langgraph(
        [*SCRIPTS["one tool, then answer"](), response(text("You're welcome!"))], registry
    )
    agent.ask("What's in Austin?")
    first_conversation = model.requests[1]["messages"]
    agent.ask("Thanks!")

    third_request = model.requests[2]["messages"]
    assert third_request[: len(first_conversation)] == first_conversation
    assert third_request[-1] == {"role": "user", "content": "Thanks!"}


def test_reset_starts_a_new_conversation(registry):
    agent, model = make_langgraph([response(text("a")), response(text("b"))], registry)
    agent.ask("first")
    agent.reset()
    agent.ask("second")
    assert model.requests[1]["messages"] == [{"role": "user", "content": "second"}]


# --- Differences the engine has to handle -------------------------------------------------------


def test_parallel_saves_all_reach_the_shortlist(registry):
    """LangGraph runs parallel tool calls on separate threads. Without the lock in
    langchain_tools.py, concurrent saves overwrite each other and entries are lost."""
    ids = [f"sample-00{i}" for i in range(1, 8)]
    agent, _ = make_langgraph(
        [
            response(
                *[tool_use(f"t{i}", "save_to_shortlist", {"event_id": e}) for i, e in enumerate(ids)],
                stop_reason="tool_use",
            ),
            response(text("Saved.")),
        ],
        registry,
    )
    agent.ask("save them all")

    saved = json.loads(registry.run("list_shortlist", {}).content)
    assert sorted(e["event_id"] for e in saved["entries"]) == ids


def test_langgraph_runs_a_tool_call_from_a_cut_off_reply_unlike_the_handwritten_loop(registry):
    """A known difference, recorded here on purpose: LangGraph routes on the presence of
    tool calls and doesn't look at stop_reason. ToolRegistry still validates the input."""

    def script():
        return [
            response(tool_use("t1", "get_event_details", {}), stop_reason="max_tokens"),
            response(text("Sorry, let me retry.")),
        ]

    handwritten, client = make_handwritten(script(), registry)
    langgraph, model = make_langgraph(script(), registry)

    assert handwritten.ask("q").stop_reason == "max_tokens"
    assert len(client.requests) == 1

    assert langgraph.ask("q").stop_reason == "end_turn"
    assert len(model.requests) == 2
    assert "event_id" in model.requests[1]["messages"][2]["content"][0]["content"]


def test_usage_conversion_separates_cached_tokens():
    usage = _usage(
        AIMessage(
            content="",
            usage_metadata={
                "input_tokens": 1500,
                "output_tokens": 40,
                "total_tokens": 1540,
                "input_token_details": {"cache_read": 1000, "cache_creation": 300},
            },
        )
    )
    assert (usage.input_tokens, usage.cache_read_input_tokens, usage.cache_creation_input_tokens) == (200, 1000, 300)
    assert usage.output_tokens == 40


def test_usage_conversion_counts_cache_writes_reported_by_lifetime():
    usage = _usage(
        AIMessage(
            content="",
            usage_metadata={
                "input_tokens": 1500,
                "output_tokens": 40,
                "total_tokens": 1540,
                "input_token_details": {
                    "cache_read": 1000,
                    "cache_creation": 0,  # LangChain zeroes this when the lifetime keys are set
                    "ephemeral_5m_input_tokens": 200,
                    "ephemeral_1h_input_tokens": 100,
                },
            },
        )
    )
    assert (usage.input_tokens, usage.cache_read_input_tokens, usage.cache_creation_input_tokens) == (200, 1000, 300)


def test_usage_conversion_handles_missing_data():
    usage = _usage(AIMessage(content=""))
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_input_tokens) == (0, 0, 0)


def test_export_messages_matches_the_api_format(registry):
    agent, _ = make_langgraph(SCRIPTS["two tools in one reply"](), registry)
    agent.ask("question")

    history = agent.export_messages()
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert sorted(r["tool_use_id"] for r in history[2]["content"]) == ["t1", "t2"]
    json.dumps(history)  # must be JSON-ready for the /history command


# --- The CLI ------------------------------------------------------------------------------------


@pytest.fixture
def isolated_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "TICKETMASTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_cli_runs_the_langgraph_engine(isolated_cli, capsys):
    model = scripted_model(SCRIPTS["one tool, then answer"]())

    assert cli.main(["--engine", "langgraph", "What's in Austin?"], chat_model=model) == 0

    out = capsys.readouterr().out
    assert 'tool call  search_events {"city": "Austin"}' in out
    assert "done in 2 steps" in out
    assert out.rstrip().endswith("Austin has shows.")
    assert "OFFLINE SAMPLE" in model.requests[0]["system"]


def test_cli_chat_commands_work_with_the_langgraph_engine(isolated_cli, monkeypatch, capsys):
    lines = iter(["hello", "/history", "/reset"])

    def fake_input(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    model = scripted_model([response(thinking("Hi."), text("Hi there!"))])

    assert cli.main(["--engine", "langgraph", "--quiet"], chat_model=model) == 0

    out = capsys.readouterr().out
    assert "Engine: langgraph." in out
    assert '"signature": "<14 chars>"' in out
    assert "Started a new conversation." in out


def test_cli_explains_how_to_install_a_missing_extra(isolated_cli, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    # A None entry in sys.modules makes `import` raise ImportError.
    monkeypatch.setitem(__import__("sys").modules, "event_scout.langgraph_agent", None)

    assert cli.main(["--engine", "langgraph", "hi"]) == 2
    assert "uv sync --extra langchain" in capsys.readouterr().err


def test_langgraph_engine_needs_an_api_key_specifically(isolated_cli):
    assert not cli.has_credentials_for("langgraph", {"ANTHROPIC_AUTH_TOKEN": "tok"})
    assert cli.has_credentials_for("langgraph", {"ANTHROPIC_API_KEY": "sk"})
    assert cli.has_credentials_for("handwritten", {"ANTHROPIC_AUTH_TOKEN": "tok"})


def test_cli_without_a_key_exits_with_instructions(isolated_cli, capsys):
    assert cli.main(["--engine", "langgraph", "hi"]) == 2
    assert "No Claude API key found" in capsys.readouterr().err
