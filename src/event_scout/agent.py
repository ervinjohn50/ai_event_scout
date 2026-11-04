"""The agent loop: the heart of the project.

An "agent" is just this loop:

    send the conversation to Claude
    -> Claude replies. Its `stop_reason` says why it stopped:
         "tool_use"  it wants us to run one or more tools
         "end_turn"  it's done and the reply is the answer
    -> if tools were requested: run them, add the results to the conversation, repeat
    -> otherwise: return the answer

Claude decides WHICH tools to call and WHEN to stop. Our code decides HOW tools run
and enforces the limits (a maximum number of steps, input validation).

Two rules the API depends on, both enforced below:
1. The API is stateless. It remembers nothing between requests, so every request
   carries the full conversation (`self.messages`). We only ever APPEND to that list.
   We never edit or delete earlier entries.
2. Every `tool_use` block Claude sends must be answered by a `tool_result` block with
   the same id, in the very next user message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import FALLBACK_BETA, AgentConfig
from .tools import ToolOutcome, ToolRegistry
from .usage import TokenUsage


class AgentObserver:
    """Hooks the loop calls as it runs. These do nothing; `trace.ConsoleTrace` prints them.

    Keeping display code out of the loop means the loop stays readable, and tests can
    plug in their own observer to check what happened.
    """

    def on_model_call(self, step: int, messages: list[dict[str, Any]]) -> None:
        pass

    def on_model_response(self, step: int, response: Any) -> None:
        pass

    def on_tool_result(self, tool_use: Any, outcome: ToolOutcome) -> None:
        pass


@dataclass
class AgentReply:
    text: str
    stop_reason: str  # the API's stop_reason for the last step, or "step_limit"
    steps: int  # how many model calls this question took
    usage: TokenUsage = field(default_factory=TokenUsage)


class ScoutAgent:
    def __init__(
        self,
        client: Any,  # an anthropic.Anthropic client (or a fake one in tests)
        tools: ToolRegistry,
        system_prompt: str,
        config: AgentConfig = AgentConfig(),
        observer: AgentObserver | None = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.system_prompt = system_prompt
        self.config = config
        self.observer = observer or AgentObserver()
        # Built once so every request sends exactly the same tool list (see prompts.py
        # for why staying identical matters).
        self.tool_definitions = tools.definitions()
        self.messages: list[dict[str, Any]] = []
        self.total_usage = TokenUsage()

    def reset(self) -> None:
        """Start a fresh conversation. The shortlist file is unaffected."""
        self.messages = []

    def ask(self, user_text: str) -> AgentReply:
        """Answer one user message, calling tools as many times as Claude needs."""
        self.messages.append({"role": "user", "content": user_text})
        usage = TokenUsage()

        for step in range(1, self.config.max_steps + 1):
            # 1. Send the whole conversation so far.
            self.observer.on_model_call(step, self.messages)
            response = self._call_model()
            usage.add(response.usage)
            self.total_usage.add(response.usage)

            # 2. Record Claude's reply exactly as received: every content block, including
            #    thinking blocks. Claude needs its earlier thinking to continue coherently,
            #    and the API rejects thinking blocks that were modified.
            self.messages.append({"role": "assistant", "content": response.content})
            self.observer.on_model_response(step, response)

            # 3. Decide what happens next, based on why Claude stopped.
            if response.stop_reason == "tool_use":
                tool_results = self._run_tools(response.content)
                # All results go back together in ONE user message.
                self.messages.append({"role": "user", "content": tool_results})
                continue

            if response.stop_reason == "pause_turn":
                # The API paused a long turn. Sending the conversation again (with no new
                # user message) lets it resume. Only server-side tools cause this; ours
                # all run locally, but handling it keeps the loop correct.
                continue

            # Anything else ends the turn: "end_turn" (finished), "max_tokens" (cut off),
            # "refusal" (declined), and so on.
            self._answer_unrun_tool_calls(response.content)
            return AgentReply(self._final_text(response), response.stop_reason, step, usage)

        # 4. Safety net: Claude never finished. Stop instead of looping (and paying) forever.
        return AgentReply(
            f"(Stopped after {self.config.max_steps} steps without a final answer. "
            "Try a more specific question.)",
            "step_limit",
            self.config.max_steps,
            usage,
        )

    # --- The three things the loop does ---------------------------------------------------

    def _call_model(self) -> Any:
        request: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.system_prompt,
            "tools": self.tool_definitions,
            "messages": self.messages,
            # Claude Opus 5.5 always thinks before it answers. "summarized" asks the API
            # to also send back a readable summary of that thinking, which the trace prints.
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": self.config.effort},
            # Automatic prompt caching: the API caches the request up to its last block, so
            # the next step (same start, plus a few new messages) is mostly a cache hit.
            "cache_control": {"type": "ephemeral"},
        }
        if self.config.use_fallbacks:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        # `client.beta.messages` (not `client.messages`) because `fallbacks` is a beta feature.
        return self.client.beta.messages.create(**request)

    def _run_tools(self, content: list[Any]) -> list[dict[str, Any]]:
        """Run every tool Claude asked for in this reply and build the tool_result blocks."""
        results = []
        for block in content:
            # Find blocks by their `type`, never by position: a reply can mix thinking,
            # text, and several tool_use blocks in any order.
            if block.type != "tool_use":
                continue
            outcome = self.tools.run(block.name, block.input)  # never raises (see tools.py)
            self.observer.on_tool_result(block, outcome)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,  # must match the tool_use block's id
                    "content": outcome.content,
                    "is_error": outcome.is_error,
                }
            )
        return results

    def _answer_unrun_tool_calls(self, content: list[Any]) -> None:
        """Keep the history valid when a reply stopped mid-tool-call.

        If a reply hit max_tokens (or was declined) while writing a tool call, that call
        may be incomplete, so we must not run it. But rule 2 still applies: the next
        request is rejected unless every tool_use has a tool_result. So we answer each
        one with an error saying it wasn't run.
        """
        unanswered = [block for block in content if block.type == "tool_use"]
        if unanswered:
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Not run: the reply stopped before this tool call was complete.",
                            "is_error": True,
                        }
                        for block in unanswered
                    ],
                }
            )

    def _final_text(self, response: Any) -> str:
        text = "\n".join(block.text for block in response.content if block.type == "text").strip()

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            reason = f" (category: {category})" if category else ""
            return (text + "\n\n" if text else "") + f"(Claude declined this request{reason}.)"

        if response.stop_reason in ("max_tokens", "model_context_window_exceeded"):
            return (text + "\n\n" if text else "") + (
                f"(The reply was cut off: stop_reason={response.stop_reason}.)"
            )

        return text or "(Claude finished without writing a reply.)"

    # --- For learning: see exactly what the API sees ---------------------------------------

    def export_messages(self) -> list[dict[str, Any]]:
        """The conversation as plain JSON-ready dicts (SDK objects converted)."""
        return [
            {"role": m["role"], "content": _to_plain(m["content"])} for m in self.messages
        ]


def _to_plain(content: Any) -> Any:
    if isinstance(content, list):
        return [_to_plain(item) for item in content]
    if hasattr(content, "model_dump"):  # an SDK content block (a Pydantic model)
        return content.model_dump(mode="json", exclude_none=True)
    return content
