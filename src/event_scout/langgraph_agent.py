"""The same agent as agent.py, built with LangChain and LangGraph.

`create_agent` builds a graph with two nodes, which is the loop from agent.py in
another form:

    START -> [model] -> did it ask for tools? -- yes --> [tools] -> back to [model]
                                              -- no ---> END

The library handles calling the model, running tools, and pairing tool_use with
tool_result. This class handles what the app needs around that: keeping the conversation
between questions, reporting each step, and translating LangChain's reporting into the
types the rest of the app uses.

Known difference from agent.py: LangGraph runs a tool call whenever a reply contains
one, even if the reply was cut off by max_tokens. agent.py refuses to run those.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from .agent import AgentObserver, AgentReply
from .config import FALLBACK_BETA, AgentConfig
from .langchain_tools import to_langchain_tools
from .tools import ToolOutcome, ToolRegistry
from .usage import TokenUsage


def build_chat_model(config: AgentConfig) -> ChatAnthropic:
    """A Claude client with the same request settings as `ScoutAgent._call_model`."""
    model_kwargs: dict[str, Any] = {"cache_control": {"type": "ephemeral"}}
    betas = None
    if config.use_fallbacks:
        model_kwargs["fallbacks"] = "default"
        betas = [FALLBACK_BETA]
    return ChatAnthropic(
        model=config.model,
        max_tokens=config.max_tokens,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": config.effort},
        betas=betas,
        model_kwargs=model_kwargs,
    )


class LangGraphScoutAgent:
    """Has the same public interface as `ScoutAgent`, so the CLI can use either one."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: ToolRegistry,
        system_prompt: str,
        config: AgentConfig = AgentConfig(),
        observer: AgentObserver | None = None,
    ) -> None:
        self.config = config
        self.observer = observer or AgentObserver()
        self.graph = create_agent(model, to_langchain_tools(tools), system_prompt=system_prompt)
        self.messages: list[BaseMessage] = []
        self.total_usage = TokenUsage()

    def reset(self) -> None:
        self.messages = []

    def ask(self, user_text: str) -> AgentReply:
        self.messages.append(HumanMessage(user_text))
        usage = TokenUsage()
        step = 1
        pending_tool_calls: dict[str, Any] = {}
        last_reply: AIMessage | None = None

        # LangGraph's limit counts graph steps, not model calls. Each model call plus its
        # tool run is two steps, so this allows exactly max_steps model calls.
        run_config = {"recursion_limit": 2 * self.config.max_steps}

        self.observer.on_model_call(step, self.export_messages())
        try:
            # "updates" yields each node's new messages as soon as that node finishes.
            for update in self.graph.stream(
                {"messages": list(self.messages)}, run_config, stream_mode="updates"
            ):
                for node, output in update.items():
                    for message in (output or {}).get("messages", []):
                        self.messages.append(message)

                        if isinstance(message, AIMessage):
                            last_reply = message
                            response = _as_api_response(message)
                            usage.add(response.usage)
                            self.total_usage.add(response.usage)
                            self.observer.on_model_response(step, response)
                            pending_tool_calls = {
                                block.id: block for block in response.content if block.type == "tool_use"
                            }

                        elif isinstance(message, ToolMessage):
                            tool_use = pending_tool_calls.pop(message.tool_call_id, None)
                            outcome = ToolOutcome(message.text, is_error=message.status == "error")
                            self.observer.on_tool_result(tool_use, outcome)
                            # Tool results arrive one at a time and in any order. The next
                            # model call only starts once all of them are in.
                            if not pending_tool_calls and step < self.config.max_steps:
                                step += 1
                                self.observer.on_model_call(step, self.export_messages())
        except GraphRecursionError:
            return AgentReply(
                f"(Stopped after {self.config.max_steps} steps without a final answer. "
                "Try a more specific question.)",
                "step_limit",
                self.config.max_steps,
                usage,
            )

        response = _as_api_response(last_reply)
        return AgentReply(_final_text(response), response.stop_reason, step, usage)

    def export_messages(self) -> list[dict[str, Any]]:
        """The conversation in the API's format, for the CLI's /history command."""
        exported: list[dict[str, Any]] = []
        for message in self.messages:
            if isinstance(message, ToolMessage):
                result = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.text,
                    "is_error": message.status == "error",
                }
                # All results from one reply share one user message, as sent to the API.
                previous = exported[-1] if exported else None
                if previous and previous["role"] == "user" and isinstance(previous["content"], list):
                    previous["content"].append(result)
                else:
                    exported.append({"role": "user", "content": [result]})
            elif isinstance(message, AIMessage):
                exported.append({"role": "assistant", "content": message.content})
            else:
                exported.append({"role": "user", "content": message.content})
        return exported


# --- Translating LangChain's reporting into the types the app uses --------------------------


def _as_api_response(message: AIMessage) -> SimpleNamespace:
    """Give an AIMessage the same shape as an API response, for the trace and the reply."""
    metadata = message.response_metadata
    details = metadata.get("stop_details")
    return SimpleNamespace(
        model=metadata.get("model_name") or metadata.get("model"),
        content=_content_blocks(message),
        stop_reason=metadata.get("stop_reason"),
        stop_details=SimpleNamespace(**details) if details else None,
        usage=_usage(message),
    )


def _content_blocks(message: AIMessage) -> list[SimpleNamespace]:
    # A reply with a single text block arrives as a plain string, anything else as a list.
    if isinstance(message.content, str):
        return [SimpleNamespace(type="text", text=message.content)] if message.content else []
    blocks = []
    for block in message.content:
        if block.get("type") == "fallback":
            blocks.append(
                SimpleNamespace(
                    type="fallback",
                    from_=SimpleNamespace(**(block.get("from") or block.get("from_"))),
                    to=SimpleNamespace(**block["to"]),
                )
            )
        else:
            blocks.append(SimpleNamespace(**block))
    return blocks


def _usage(message: AIMessage) -> SimpleNamespace:
    """Convert LangChain's token counts to the API's.

    The API reports uncached input tokens separately from cache reads and writes, while
    LangChain's `input_tokens` is the total of all three. Cache writes need care: when
    the API reports them by cache lifetime, LangChain sets `cache_creation` to 0 and
    keeps the counts under the lifetime keys.
    """
    usage = message.usage_metadata or {}
    details = usage.get("input_token_details") or {}
    cache_read = details.get("cache_read") or 0
    cache_write = (
        (details.get("cache_creation") or 0)
        + (details.get("ephemeral_5m_input_tokens") or 0)
        + (details.get("ephemeral_1h_input_tokens") or 0)
    )
    return SimpleNamespace(
        input_tokens=max(0, usage.get("input_tokens", 0) - cache_read - cache_write),
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        output_tokens=usage.get("output_tokens", 0),
    )


def _final_text(response: SimpleNamespace) -> str:
    """Same wording as `ScoutAgent._final_text`. A test checks that the two stay in sync."""
    text = "\n".join(block.text for block in response.content if block.type == "text").strip()

    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        reason = f" (category: {category})" if category else ""
        return (text + "\n\n" if text else "") + f"(Claude declined this request{reason}.)"

    if response.stop_reason in ("max_tokens", "model_context_window_exceeded"):
        return (text + "\n\n" if text else "") + (
            f"(The reply was cut off: stop_reason={response.stop_reason}.)"
        )

    return text or "(Claude finished without writing a reply.)"
