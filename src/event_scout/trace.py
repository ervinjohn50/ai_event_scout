"""Prints each step of the agent loop as it runs, so the agent's work is visible.

A typical question looks like this:

    -- step 1 -------------------------------------- sending 1 message
      thinking   The user wants indie rock in Chicago within two weeks...
      tool call  search_events {"city": "Chicago", "keyword": "indie rock", ...}
      result     ok, 412 chars: {"count": 2, "events": [...]}
      usage      stop_reason=tool_use | prompt 2,140 (0 cached) | output 180 | ~$0.0124
    -- step 2 -------------------------------------- sending 3 messages
      ...
      usage      stop_reason=end_turn | prompt 2,690 (2,140 cached) | ...
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any, TextIO

from .agent import AgentObserver
from .config import Pricing
from .tools import ToolOutcome
from .usage import TokenUsage

LABEL_WIDTH = 11
PREVIEW_CHARS = 300  # how much of thinking / tool results to show unless verbose


class ConsoleTrace(AgentObserver):
    def __init__(
        self,
        pricing: Pricing | None = None,
        verbose: bool = False,
        out: TextIO | None = None,
        color: bool | None = None,
    ) -> None:
        self.pricing = pricing
        self.verbose = verbose
        # Not `out: TextIO = sys.stdout`: a default value is evaluated once, when the
        # module is imported, so it would miss any later redirect of sys.stdout.
        self._out = out
        if color is None:  # color only in a real terminal, and never if NO_COLOR is set
            color = self.out.isatty() and "NO_COLOR" not in os.environ
        self.color = color

    @property
    def out(self) -> TextIO:
        return self._out or sys.stdout

    # --- AgentObserver hooks ---------------------------------------------------------------

    def on_model_call(self, step: int, messages: list[dict[str, Any]]) -> None:
        n = len(messages)
        right = f" sending {n} message{'s' if n != 1 else ''} "
        left = f"-- step {step} "
        self._print(self._style(left + "-" * max(4, 60 - len(left) - len(right)) + right, "yellow"))

    def on_model_response(self, step: int, response: Any) -> None:
        for block in response.content:
            if block.type == "thinking":
                if block.thinking:
                    self._line("thinking", self._shorten(block.thinking), "dim")
            elif block.type == "redacted_thinking":
                self._line("thinking", "(redacted)", "dim")
            elif block.type == "text" and response.stop_reason == "tool_use":
                # Text written before a tool call. Final answers are printed by the CLI.
                self._line("says", self._shorten(block.text), "dim")
            elif block.type == "tool_use":
                args = json.dumps(block.input, ensure_ascii=False)
                self._line("tool call", f"{block.name} {args}", "cyan")
            elif block.type == "fallback":
                self._line("fallback", f"{block.from_.model} declined; {block.to.model} continued", "yellow")

        usage = TokenUsage.from_api(response.usage)
        parts = [
            f"stop_reason={response.stop_reason}",
            f"prompt {usage.prompt_tokens:,} ({usage.cache_read_tokens:,} cached)",
            f"output {usage.output_tokens:,}",
        ]
        cost = usage.cost(self.pricing)
        if cost is not None:
            parts.append(f"~${cost:.4f}")
        self._line("usage", " | ".join(parts), "dim")

    def on_tool_result(self, tool_use: Any, outcome: ToolOutcome) -> None:
        if outcome.is_error:
            self._line("result", f"ERROR: {outcome.content}", "red")
        else:
            self._line("result", f"ok, {len(outcome.content):,} chars: {self._shorten(outcome.content)}", "green")

    # --- Summary after each question ---------------------------------------------------------

    def print_turn_summary(self, steps: int, usage: TokenUsage) -> None:
        cost = usage.cost(self.pricing)
        cost_text = f", ~${cost:.4f}" if cost is not None else ""
        self._print(
            self._style(
                f"-- done in {steps} step{'s' if steps != 1 else ''}: "
                f"{usage.prompt_tokens:,} prompt tokens ({usage.cache_read_tokens:,} cached), "
                f"{usage.output_tokens:,} output tokens{cost_text}",
                "yellow",
            )
        )

    # --- Formatting helpers ------------------------------------------------------------------

    def _line(self, label: str, text: str, style: str) -> None:
        wrapped = textwrap.fill(
            " ".join(text.split()),  # collapse newlines so each entry is one paragraph
            width=100,
            initial_indent="  " + label.ljust(LABEL_WIDTH),
            subsequent_indent=" " * (LABEL_WIDTH + 2),
            break_on_hyphens=False,
        )
        self._print(self._style(wrapped, style))

    def _shorten(self, text: str) -> str:
        if self.verbose or len(text) <= PREVIEW_CHARS:
            return text
        return text[:PREVIEW_CHARS].rstrip() + " ..."

    _CODES = {"dim": "2", "red": "31", "green": "32", "yellow": "33", "cyan": "36"}

    def _style(self, text: str, style: str) -> str:
        if not self.color:
            return text
        return f"\033[{self._CODES[style]}m{text}\033[0m"

    def _print(self, text: str) -> None:
        print(text, file=self.out, flush=True)
