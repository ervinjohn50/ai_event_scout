"""The terminal chat: wires everything together and talks to the user.

Run it with `uv run event-scout` (chat) or `uv run event-scout "your question"` (ask once).
This file builds the pieces (data source, tools, system prompt, agent, trace) and
handles everything around the agent: commands, API errors, and Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import anthropic

from . import __version__
from .agent import AgentObserver, AgentReply, ScoutAgent
from .config import DEFAULT_EFFORT, DEFAULT_MAX_STEPS, DEFAULT_MODEL, EFFORT_LEVELS, PRICING, AgentConfig
from .events import EventSource, SampleEventSource, TicketmasterEventSource
from .prompts import build_system_prompt
from .shortlist import DEFAULT_SHORTLIST_PATH, ShortlistError, ShortlistStore
from .tools import ToolRegistry, build_tools
from .trace import ConsoleTrace
from .tracing import configure_tracing, flush_traces

ENGINES = ("handwritten", "langgraph")


class Engine(Protocol):
    """What the CLI needs from an agent. `ScoutAgent` and `LangGraphScoutAgent` both fit."""

    config: AgentConfig

    def ask(self, user_text: str) -> AgentReply: ...

    def reset(self) -> None: ...

    def export_messages(self) -> list[dict[str, Any]]: ...


HELP_TEXT = """\
Commands:
  /shortlist  show your saved events (no AI call)
  /history    print the raw conversation the API sees, as JSON
  /reset      start a new conversation (your shortlist is kept)
  /help       show this help
  /quit       exit (Ctrl-D also works)"""

NO_CREDENTIALS_TEXT = """\
No Claude API key found.

  1. Create a key at https://platform.claude.com/settings/keys
  2. Copy .env.example to .env and set ANTHROPIC_API_KEY=... in it
     (or run: export ANTHROPIC_API_KEY=...)

The offline tests don't need a key: uv run pytest"""

LANGCHAIN_MISSING_TEXT = """\
The langgraph engine needs the optional LangChain packages. Install them with:

  uv sync --extra langchain"""


def main(argv: list[str] | None = None, client: Any = None, chat_model: Any = None) -> int:
    """Entry point. `client` and `chat_model` let tests pass in fakes instead of real ones."""
    load_dotenv(Path(".env"))
    args = parse_args(argv)

    if client is None and chat_model is None and not has_credentials_for(args.engine, os.environ):
        print(NO_CREDENTIALS_TEXT, file=sys.stderr)
        return 2

    source = choose_source(args.offline, os.environ)
    store = ShortlistStore(args.shortlist_path)
    config = AgentConfig(
        model=args.model,
        effort=args.effort,
        max_steps=args.max_steps,
        use_fallbacks=not args.no_fallbacks,
    )
    trace = None if args.quiet else ConsoleTrace(pricing=PRICING.get(args.model), verbose=args.verbose)
    tracing = configure_tracing()
    try:
        agent = build_agent(
            args.engine,
            build_tools(source, store),
            build_system_prompt(date.today(), source.describe()),
            config,
            trace,
            client=client,
            chat_model=chat_model,
            tracing=tracing,
        )
    except ImportError:
        print(LANGCHAIN_MISSING_TEXT, file=sys.stderr)
        return 2

    try:
        if args.question:
            return 0 if ask_and_print(agent, trace, args.question) else 1
        print_banner(config, source, args.engine, tracing)
        return repl(agent, trace, store)
    finally:
        if tracing:
            flush_traces()


def build_agent(
    engine: str,
    tools: ToolRegistry,
    system_prompt: str,
    config: AgentConfig,
    observer: AgentObserver | None,
    client: Any = None,
    chat_model: Any = None,
    tracing: bool = False,
) -> Engine:
    """Create the chosen engine. Both get the same tools, prompt, settings, and observer."""
    agent: Engine
    if engine == "langgraph":
        # Imported here because LangChain is optional: the handwritten engine must work
        # without it. A missing package raises ImportError, which main() reports.
        from .langgraph_agent import LangGraphScoutAgent, build_chat_model

        agent = LangGraphScoutAgent(chat_model or build_chat_model(config), tools, system_prompt, config, observer)
    else:
        client = client or anthropic.Anthropic()
        if tracing:
            from .tracing import TracedToolRegistry, trace_client

            client, tools = trace_client(client), TracedToolRegistry(tools)
        agent = ScoutAgent(client, tools, system_prompt, config, observer)

    if tracing:
        from .tracing import TracedEngine

        agent = TracedEngine(agent, engine)
    return agent


# --- Setup helpers -----------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="event-scout",
        description="Chat with an AI agent that finds concerts and live events.",
    )
    parser.add_argument("question", nargs="?", help="ask one question and exit (omit to chat)")
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default="handwritten",
        help="which agent loop to run: this project's own, or LangGraph (default: handwritten)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model id (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--effort",
        choices=EFFORT_LEVELS,
        default=DEFAULT_EFFORT,
        help=f"how hard the model thinks; lower is faster and cheaper (default: {DEFAULT_EFFORT})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"most model calls per question (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--offline", action="store_true", help="use the bundled sample events even if a Ticketmaster key is set"
    )
    parser.add_argument("--no-fallbacks", action="store_true", help="don't send the refusal-fallback beta")
    parser.add_argument(
        "--shortlist-path",
        type=Path,
        default=DEFAULT_SHORTLIST_PATH,
        help=f"where saved events are stored (default: {DEFAULT_SHORTLIST_PATH})",
    )
    detail = parser.add_mutually_exclusive_group()
    detail.add_argument("--quiet", action="store_true", help="hide the step-by-step trace")
    detail.add_argument("--verbose", action="store_true", help="show full thinking and tool results")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file into the environment.

    A tiny version of the python-dotenv package. Variables that are already set win,
    so `export ANTHROPIC_API_KEY=...` in your shell overrides the file.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def has_anthropic_credentials(env: Mapping[str, str]) -> bool:
    """Check for credentials up front.

    Without this, the SDK only fails when the first request is sent, with a
    confusing TypeError. Failing early with instructions is friendlier.
    """
    if env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_PROFILE"):
        return True
    # `ant auth login` (Anthropic's CLI) stores a profile here that the SDK also reads.
    return (Path.home() / ".config" / "anthropic").is_dir()


def has_credentials_for(engine: str, env: Mapping[str, str]) -> bool:
    # ChatAnthropic reads only ANTHROPIC_API_KEY, not the SDK's other credential sources.
    if engine == "langgraph":
        return bool(env.get("ANTHROPIC_API_KEY"))
    return has_anthropic_credentials(env)


def choose_source(offline: bool, env: Mapping[str, str]) -> EventSource:
    api_key = env.get("TICKETMASTER_API_KEY")
    if api_key and not offline:
        return TicketmasterEventSource(api_key)
    return SampleEventSource()


def print_banner(config: AgentConfig, source: EventSource, engine: str, tracing: bool = False) -> None:
    data = "live Ticketmaster data" if source.name == "ticketmaster" else (
        "offline sample data (set TICKETMASTER_API_KEY for live events)"
    )
    print(f"Event Scout | {config.model}, effort {config.effort} | {data}")
    print(f"Engine: {engine}. LangSmith tracing: {'on' if tracing else 'off'}.")
    print("Ask about concerts and events, or type /help.")


# --- Asking questions ----------------------------------------------------------------------


def ask_and_print(agent: Engine, trace: ConsoleTrace | None, question: str) -> bool:
    """Run one question through the agent and print the answer. Returns False on failure.

    API errors are caught most-specific first, so each gets a message that says what
    to do about it. All of them are subclasses of APIStatusError, except connection errors.
    """
    try:
        reply = agent.ask(question)
    except anthropic.AuthenticationError:
        return _fail("The API key was rejected. Check ANTHROPIC_API_KEY.")
    except anthropic.PermissionDeniedError:
        return _fail("This API key doesn't have access to that model or feature.")
    except anthropic.NotFoundError:
        return _fail(f"Model '{agent.config.model}' wasn't found. Check the --model value.")
    except anthropic.RateLimitError:
        return _fail("Rate limited by the API. Wait a minute and try again.")
    except anthropic.BadRequestError as err:
        return _fail(f"The API rejected the request: {err.message}\n(If this repeats, type /reset.)")
    except anthropic.APIStatusError as err:
        return _fail(f"The API returned an error (HTTP {err.status_code}). Try again shortly.")
    except anthropic.APIConnectionError:
        return _fail("Couldn't reach the Claude API. Check your internet connection.")

    if trace:
        trace.print_turn_summary(reply.steps, reply.usage)
    print(f"\n{reply.text}")
    return True


def _fail(message: str) -> bool:
    print(f"\nError: {message}", file=sys.stderr)
    return False


# --- Interactive chat ----------------------------------------------------------------------


def repl(agent: Engine, trace: ConsoleTrace | None, store: ShortlistStore) -> int:
    """Read-eval-print loop: read a line, act on it, print the result, repeat."""
    try:
        import readline  # noqa: F401  (gives input() arrow keys and history on macOS/Linux)
    except ImportError:
        pass

    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl-D or Ctrl-C at the prompt
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            if not handle_command(line, agent, store):
                return 0
            continue
        try:
            ask_and_print(agent, trace, line)
        except KeyboardInterrupt:
            # Ctrl-C mid-answer can leave a tool call without its result, and the API
            # rejects that history. Starting over is the simple, safe fix.
            agent.reset()
            print("\n(Interrupted. Started a new conversation.)")


def handle_command(line: str, agent: Engine, store: ShortlistStore) -> bool:
    """Handle a /command. Returns False when the user wants to quit."""
    command = line.split()[0].lower()
    if command in ("/quit", "/exit"):
        return False
    if command == "/help":
        print(HELP_TEXT)
    elif command == "/reset":
        agent.reset()
        print("Started a new conversation.")
    elif command == "/history":
        print(json.dumps(_shorten_signatures(agent.export_messages()), indent=2, ensure_ascii=False))
    elif command == "/shortlist":
        print_shortlist(store)
    else:
        print(f"Unknown command {command}. Type /help.")
    return True


def print_shortlist(store: ShortlistStore) -> None:
    try:
        entries = store.all()
    except ShortlistError as err:
        print(f"Error: {err}")
        return
    if not entries:
        print("Your shortlist is empty. Ask the scout to save an event.")
        return
    for entry in entries:
        place = ", ".join(part for part in (entry.get("venue"), entry.get("city")) if part)
        line = f"- {entry['date']}  {entry['name']}" + (f"  @ {place}" if place else "")
        if entry.get("price"):
            line += f"  ({entry['price']})"
        if entry.get("note"):
            line += f"  note: {entry['note']}"
        print(line)


def _shorten_signatures(value: Any) -> Any:
    """Thinking blocks carry a long opaque `signature`; abbreviate it for display only."""
    if isinstance(value, list):
        return [_shorten_signatures(item) for item in value]
    if isinstance(value, dict):
        return {
            key: (f"<{len(item)} chars>" if key == "signature" and isinstance(item, str) else _shorten_signatures(item))
            for key, item in value.items()
        }
    return value
