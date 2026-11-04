"""Tests for the console trace output."""

import io

from fakes import response, text, thinking, tool_use

from event_scout.config import PRICING
from event_scout.tools import ToolOutcome
from event_scout.trace import ConsoleTrace
from event_scout.usage import TokenUsage


def make_trace(**kwargs) -> tuple[ConsoleTrace, io.StringIO]:
    out = io.StringIO()
    return ConsoleTrace(pricing=PRICING["claude-opus-5-5"], out=out, **kwargs), out


def test_trace_shows_thinking_tool_calls_results_and_usage():
    trace, out = make_trace()
    resp = response(
        thinking("They want jazz in Chicago."),
        tool_use("t1", "search_events", {"city": "Chicago"}),
        stop_reason="tool_use",
        input_tokens=500,
        cache_read=1500,
        output_tokens=100,
    )

    trace.on_model_call(1, [{"role": "user", "content": "jazz?"}])
    trace.on_model_response(1, resp)
    trace.on_tool_result(resp.content[1], ToolOutcome('{"count": 1}'))
    trace.on_tool_result(resp.content[1], ToolOutcome("Invalid input", is_error=True))

    output = out.getvalue()
    assert "-- step 1" in output and "sending 1 message" in output
    assert "thinking   They want jazz in Chicago." in output
    assert 'tool call  search_events {"city": "Chicago"}' in output
    assert 'result     ok, 12 chars: {"count": 1}' in output
    assert "result     ERROR: Invalid input" in output
    assert "stop_reason=tool_use | prompt 2,000 (1,500 cached) | output 100 | ~$0.0043" in output
    assert "\033[" not in output  # no color codes when not writing to a terminal


def test_final_answer_text_is_left_for_the_cli_to_print():
    trace, out = make_trace()
    trace.on_model_response(1, response(text("Here are your shows."), stop_reason="end_turn"))
    assert "Here are your shows." not in out.getvalue()


def test_long_results_are_shortened_unless_verbose():
    long_result = ToolOutcome("x" * 1000)
    block = response(tool_use("t1", "search_events", {}), stop_reason="tool_use").content[0]

    trace, out = make_trace()
    trace.on_tool_result(block, long_result)
    assert "..." in out.getvalue() and "x" * 1000 not in out.getvalue()

    trace, out = make_trace(verbose=True)
    trace.on_tool_result(block, long_result)
    assert "x" * 100 in out.getvalue().replace("\n", "").replace(" ", "")


def test_color_codes_when_enabled():
    trace, out = make_trace(color=True)
    trace.on_tool_result(None, ToolOutcome("boom", is_error=True))
    assert out.getvalue().startswith("\033[31m")


def test_turn_summary():
    trace, out = make_trace()
    usage = TokenUsage(input_tokens=1000, cache_read_tokens=2000, output_tokens=300, requests=2)
    trace.print_turn_summary(2, usage)
    assert "done in 2 steps: 3,000 prompt tokens (2,000 cached), 300 output tokens, ~$0.0104" in out.getvalue()
