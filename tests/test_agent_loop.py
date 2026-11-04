"""Tests for the agent loop, using a fake Claude that replays scripted responses.

Each test reads like a little story: "Claude asks for this tool, then says that",
followed by checks on what our loop sent, ran, and returned.
"""

import json
from datetime import date

import pytest
from fakes import FakeClient, response, text, thinking, tool_use

from event_scout.agent import AgentObserver, ScoutAgent
from event_scout.config import FALLBACK_BETA, PRICING, AgentConfig
from event_scout.events import SampleEventSource
from event_scout.shortlist import ShortlistStore
from event_scout.tools import ToolRegistry, build_tools

TODAY = date(2026, 11, 3)
SYSTEM = "You are a test scout."


class SpyRegistry(ToolRegistry):
    """A real registry that also remembers which tools were run."""

    def __init__(self, inner: ToolRegistry) -> None:
        self._tools = inner._tools
        self.calls: list[tuple[str, dict]] = []

    def run(self, name, raw_input):
        self.calls.append((name, raw_input))
        return super().run(name, raw_input)


@pytest.fixture
def tools(tmp_path) -> SpyRegistry:
    return SpyRegistry(build_tools(SampleEventSource(today=TODAY), ShortlistStore(tmp_path / "s.json")))


def make_agent(responses, tools, **config) -> tuple[ScoutAgent, FakeClient]:
    client = FakeClient(responses)
    return ScoutAgent(client, tools, SYSTEM, AgentConfig(**config)), client


def tool_results_in(message: dict) -> list[dict]:
    return [block for block in message["content"] if block["type"] == "tool_result"]


# --- The basic loop ----------------------------------------------------------------------


def test_tool_call_then_final_answer(tools):
    agent, client = make_agent(
        [
            response(
                thinking("The user wants sports in Chicago."),
                tool_use("toolu_1", "search_events", {"city": "Chicago", "category": "sports"}),
                stop_reason="tool_use",
            ),
            response(text("There's one game: Prairie Wolves vs. Lakeshore Hawks."), stop_reason="end_turn"),
        ],
        tools,
    )

    reply = agent.ask("Any sports in Chicago?")

    assert reply.text == "There's one game: Prairie Wolves vs. Lakeshore Hawks."
    assert reply.stop_reason == "end_turn"
    assert reply.steps == 2
    assert tools.calls == [("search_events", {"city": "Chicago", "category": "sports"})]

    # The second request carries the whole conversation: question, Claude's tool call,
    # and our tool result answering it with the matching id.
    sent = client.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    [result] = tool_results_in(sent[2])
    assert result["tool_use_id"] == "toolu_1"
    assert result["is_error"] is False
    assert json.loads(result["content"])["events"][0]["id"] == "sample-003"

    # Final history: question, tool call, tool result, answer.
    assert [m["role"] for m in agent.messages] == ["user", "assistant", "user", "assistant"]


def test_parallel_tool_calls_are_answered_in_one_message(tools):
    agent, client = make_agent(
        [
            response(
                tool_use("toolu_a", "search_events", {"city": "Chicago"}),
                tool_use("toolu_b", "search_events", {"city": "Austin"}),
                stop_reason="tool_use",
            ),
            response(text("Here's both cities."), stop_reason="end_turn"),
        ],
        tools,
    )

    agent.ask("Chicago or Austin?")

    sent = client.requests[1]["messages"]
    assert len(sent) == 3  # ONE user message holds both results
    results = tool_results_in(sent[2])
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]


def test_tool_errors_go_back_to_claude_and_the_loop_continues(tools):
    agent, client = make_agent(
        [
            response(
                tool_use("toolu_1", "buy_tickets", {"event_id": "sample-001"}),
                tool_use("toolu_2", "search_events", {"max_results": 500}),
                stop_reason="tool_use",
            ),
            response(tool_use("toolu_3", "search_events", {"max_results": 5}), stop_reason="tool_use"),
            response(text("Fixed it."), stop_reason="end_turn"),
        ],
        tools,
    )

    reply = agent.ask("Find me something")

    assert reply.text == "Fixed it." and reply.steps == 3
    first_results = tool_results_in(client.requests[1]["messages"][2])
    assert [r["is_error"] for r in first_results] == [True, True]
    assert "Unknown tool 'buy_tickets'" in first_results[0]["content"]
    assert "max_results" in first_results[1]["content"]


# --- What exactly is sent to the API ------------------------------------------------------


def test_request_contains_model_settings_tools_and_system_prompt(tools):
    agent, client = make_agent([response(text("Hi!"))], tools, effort="low")

    agent.ask("hello")

    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["system"] == SYSTEM
    assert request["tools"] == tools.definitions()
    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert request["output_config"] == {"effort": "low"}
    assert request["max_tokens"] == 16_000
    assert request["cache_control"] == {"type": "ephemeral"}
    assert request["betas"] == [FALLBACK_BETA]
    assert request["fallbacks"] == "default"
    assert "tool_choice" not in request  # forcing a tool call is rejected on Opus 5.5


def test_fallbacks_can_be_turned_off(tools):
    agent, client = make_agent([response(text("Hi!"))], tools, use_fallbacks=False)
    agent.ask("hello")
    assert "fallbacks" not in client.requests[0] and "betas" not in client.requests[0]


def test_history_is_append_only_across_questions(tools):
    agent, client = make_agent(
        [
            response(tool_use("toolu_1", "search_events", {"city": "Austin"}), stop_reason="tool_use"),
            response(text("Austin has shows.")),
            response(text("You're welcome!")),
        ],
        tools,
    )

    agent.ask("What's in Austin?")
    history_after_first_question = agent.export_messages()
    agent.ask("Thanks!")

    # The second question's request starts with the first conversation, unchanged.
    third_request = client.requests[2]["messages"]
    assert len(third_request) == 5
    assert agent.export_messages()[:4] == history_after_first_question
    assert third_request[-1] == {"role": "user", "content": "Thanks!"}


# --- Stopping conditions ------------------------------------------------------------------


def test_step_limit_stops_a_runaway_loop(tools):
    endless = [
        response(tool_use(f"toolu_{i}", "list_shortlist", {}), stop_reason="tool_use") for i in range(10)
    ]
    agent, client = make_agent(endless, tools, max_steps=3)

    reply = agent.ask("loop forever")

    assert reply.stop_reason == "step_limit"
    assert len(client.requests) == 3
    assert "Stopped after 3 steps" in reply.text


def test_max_tokens_does_not_run_the_cut_off_tool_call(tools):
    agent, _ = make_agent(
        [
            response(
                text("Let me search"),
                tool_use("toolu_1", "search_events", {"city": "Chi"}),  # possibly truncated
                stop_reason="max_tokens",
            )
        ],
        tools,
    )

    reply = agent.ask("Anything in Chicago?")

    assert tools.calls == []  # never run a tool call that may be incomplete
    assert reply.stop_reason == "max_tokens"
    assert "cut off" in reply.text
    # ...but the call still gets a tool_result, or the next request would be rejected.
    [closing] = tool_results_in(agent.messages[-1])
    assert closing["tool_use_id"] == "toolu_1" and closing["is_error"] is True


def test_refusal_reports_the_category(tools):
    agent, _ = make_agent(
        [response(stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"})], tools
    )
    reply = agent.ask("something declined")
    assert reply.stop_reason == "refusal"
    assert "declined" in reply.text and "cyber" in reply.text


def test_pause_turn_resends_without_a_new_user_message(tools):
    agent, client = make_agent(
        [response(text("Working..."), stop_reason="pause_turn"), response(text("Done."))], tools
    )

    reply = agent.ask("go")

    assert reply.text == "Done."
    assert [m["role"] for m in client.requests[1]["messages"]] == ["user", "assistant"]


# --- Usage and observers ------------------------------------------------------------------


def test_usage_is_summed_across_steps(tools):
    agent, _ = make_agent(
        [
            response(tool_use("t1", "list_shortlist", {}), stop_reason="tool_use", input_tokens=1000, output_tokens=50),
            response(text("Empty."), input_tokens=200, output_tokens=30, cache_read=1000),
        ],
        tools,
    )

    usage = agent.ask("my list?").usage

    assert (usage.input_tokens, usage.cache_read_tokens, usage.output_tokens) == (1200, 1000, 80)
    assert usage.requests == 2
    # 1200 * $4 + 1000 * $0.20 + 80 * $20, per million tokens
    assert usage.cost(PRICING["claude-opus-5-5"]) == pytest.approx(0.0066)
    assert agent.total_usage.requests == 2


def test_observer_sees_every_step_in_order(tools):
    events = []

    class Recorder(AgentObserver):
        def on_model_call(self, step, messages):
            events.append(("call", step, len(messages)))

        def on_model_response(self, step, resp):
            events.append(("response", step, resp.stop_reason))

        def on_tool_result(self, block, outcome):
            events.append(("tool", block.name, outcome.is_error))

    client = FakeClient(
        [response(tool_use("t1", "list_shortlist", {}), stop_reason="tool_use"), response(text("ok"))]
    )
    ScoutAgent(client, tools, SYSTEM, observer=Recorder()).ask("hi")

    assert events == [
        ("call", 1, 1),
        ("response", 1, "tool_use"),
        ("tool", "list_shortlist", False),
        ("call", 2, 3),
        ("response", 2, "end_turn"),
    ]


def test_reset_starts_a_new_conversation(tools):
    agent, client = make_agent([response(text("a")), response(text("b"))], tools)
    agent.ask("first")
    agent.reset()
    agent.ask("second")
    assert client.requests[1]["messages"] == [{"role": "user", "content": "second"}]
