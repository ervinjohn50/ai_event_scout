"""A fake Claude for tests: it replays scripted responses and records every request.

Why fake it? Tests that call the real API cost money, are slow, need a key, and give
different answers each run. The loop's job is to react correctly to whatever comes
back, and that can be tested with scripted responses.

Responses are real `BetaMessage` objects from the SDK, the same type the real client
returns, so the agent is tested against the actual response shape.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from anthropic.types.beta import BetaMessage


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def thinking(summary: str) -> dict[str, Any]:
    return {"type": "thinking", "thinking": summary, "signature": "fake-signature"}


def tool_use(block_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": block_id, "name": name, "input": tool_input}


def response(
    *blocks: dict[str, Any],
    stop_reason: str = "end_turn",
    input_tokens: int = 1000,
    output_tokens: int = 100,
    cache_read: int = 0,
    stop_details: dict[str, Any] | None = None,
) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_fake",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": list(blocks),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": 0,
            },
        }
    )


class FakeClient:
    """Stands in for `anthropic.Anthropic()`. Only `client.beta.messages.create` exists."""

    def __init__(self, responses: list[BetaMessage]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> BetaMessage:
        # Deep copy: the agent keeps appending to its `messages` list after this call,
        # and the test needs a snapshot of what was actually sent at this moment.
        self.requests.append(copy.deepcopy(request))
        if not self._responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        return self._responses.pop(0)
