"""A scripted Claude for the LangGraph engine.

This runs the real ChatAnthropic code (building the request, parsing the response) and
replaces only `_create`, the method that makes the network call. It replays the same
scripted responses as `fakes.FakeClient`, so both engines can be tested on one script.

`_create` is a private method of langchain-anthropic, so a future version could rename
it. If these tests start failing after an upgrade, check there first.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from anthropic.types.beta import BetaMessage
from langchain_anthropic import ChatAnthropic

from event_scout.config import AgentConfig
from event_scout.langgraph_agent import build_chat_model


class ScriptedChatAnthropic(ChatAnthropic):
    script: list[Any] = []
    requests: list[dict[str, Any]] = []

    def _create(self, payload: dict[str, Any]) -> Any:
        self.requests.append(copy.deepcopy(payload))
        if not self.script:
            raise AssertionError("ScriptedChatAnthropic ran out of scripted responses")
        message = self.script.pop(0)
        return SimpleNamespace(parse=lambda: message)


def scripted_model(responses: list[BetaMessage], config: AgentConfig = AgentConfig()) -> ScriptedChatAnthropic:
    """A scripted model with exactly the settings `build_chat_model` gives the real one."""
    real = build_chat_model(config)
    return ScriptedChatAnthropic(
        model=real.model,
        max_tokens=real.max_tokens,
        thinking=real.thinking,
        output_config=real.output_config,
        betas=real.betas,
        model_kwargs=real.model_kwargs,
        anthropic_api_key="fake-key",
        script=list(responses),
        requests=[],
    )
