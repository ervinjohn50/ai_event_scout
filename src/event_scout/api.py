"""The web API: the same agents as the CLI, reachable over HTTP.

Interactive documentation is generated at /docs while the server is running.

Endpoints are plain `def` functions, not `async def`, because the agents block while they
wait for Claude. FastAPI runs plain functions on a pool of worker threads, so one slow
question doesn't stop other requests from being answered.

Conversations live in memory, because the API is stateless and each engine keeps its own
history format. Restarting the server clears them. Saved events and run stats are in the
database and remain.

POST /api/chat/stream reports each step while the agent works, using Server-Sent Events:

    worker thread: runs the agent -> observer hooks -> queue -> stream() -> browser

The agent blocks until it's done, so it needs its own thread for updates to go out in the
meantime. The queue is the safe way to pass them between threads. If the browser disconnects,
the agent still finishes, because the tokens are already being paid for and stopping partway
could leave the conversation's history incomplete.

This server has no login, so it's only meant to run on the user's own computer.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import queue
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

import anthropic
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, StringConstraints

from .agent import AgentObserver
from .cli import (
    ENGINES,
    LANGCHAIN_MISSING_TEXT,
    NO_CREDENTIALS_TEXT,
    Engine,
    build_agent,
    describe_api_error,
    has_credentials_for,
)
from .config import PRICING, AgentConfig
from .db import DEFAULT_DB_PATH
from .events import EventSource
from .prompts import build_system_prompt
from .runs import ERROR, QuestionRun, RunLog
from .shortlist import ShortlistError, ShortlistStore
from .tools import ToolOutcome, build_tools
from .tracing import flush_traces
from .usage import TokenUsage

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4_000
MAX_SESSIONS = 100

# A comment sent during long silences, so connections aren't dropped as idle. It also means
# the stream notices a disconnected browser soon after it happens.
KEEP_ALIVE_SECONDS = 15

# Rejects requests addressed to any other host name, which stops a malicious website from
# pointing its own domain at this computer to reach the server (DNS rebinding).
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]

EngineName = Literal["custom", "langgraph"]


# --- Request and response shapes -------------------------------------------------------------


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_LENGTH)]
    session_id: str | None = None  # leave out to start a new conversation
    engine: EngineName | None = None  # applies to a new conversation; defaults to "custom"


class ToolStep(BaseModel):
    tool: str
    input: dict[str, Any]
    is_error: bool


class Usage(BaseModel):
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int


class ChatResponse(BaseModel):
    session_id: str
    engine: EngineName
    answer: str
    stop_reason: str
    steps: int
    tool_calls: list[ToolStep]
    usage: Usage
    cost_usd: float | None
    seconds: float


class ShortlistEntry(BaseModel):
    event_id: str
    name: str
    date: str
    venue: str | None
    city: str | None
    price: str | None
    ticket_url: str | None
    source: str
    note: str | None
    saved_at: str


class ShortlistResponse(BaseModel):
    entries: list[ShortlistEntry]


class EngineStats(BaseModel):
    engine: str
    questions: int
    errors: int
    avg_steps: float | None
    avg_tool_calls: float | None
    avg_seconds: float | None
    avg_prompt_tokens: float | None
    avg_output_tokens: float | None
    total_cost_usd: float


class StatsResponse(BaseModel):
    engines: list[EngineStats]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    data_source: str
    model: str
    available_engines: list[str]
    tracing: bool


# --- Conversations ---------------------------------------------------------------------------


Publish = Callable[[str, dict[str, Any]], None]


class _StepRecorder(AgentObserver):
    """Collects the tool calls for one question, and reports progress if anyone's listening.

    Only tool names and inputs are reported. Thinking text and tool results stay on the server.
    """

    def __init__(self) -> None:
        self.steps: list[ToolStep] = []
        self.publish: Publish | None = None

    def _report(self, event: str, payload: dict[str, Any]) -> None:
        if self.publish is not None:
            self.publish(event, payload)

    def on_model_call(self, step: int, messages: list[dict[str, Any]]) -> None:
        self._report("model_call", {"step": step})

    def on_model_response(self, step: int, response: Any) -> None:
        for block in response.content:
            if block.type == "tool_use":
                self._report("tool_start", {"id": block.id, "tool": block.name, "input": dict(block.input)})

    def on_tool_result(self, tool_use: Any, outcome: ToolOutcome) -> None:
        step = ToolStep(tool=tool_use.name, input=dict(tool_use.input), is_error=outcome.is_error)
        self.steps.append(step)
        self._report("tool_result", {"id": tool_use.id, **step.model_dump()})


@dataclass
class _Session:
    id: str
    engine: EngineName
    agent: Engine
    recorder: _StepRecorder
    # Two requests at once could break the tool call and result pairing in the history,
    # so a second question in the same conversation waits for the first to finish.
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False  # a request waiting on the lock checks this once it gets through


class _SessionStore:
    """Conversations in memory, up to a limit. The least recently used one is dropped first."""

    def __init__(self, max_sessions: int) -> None:
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._guard = threading.Lock()

    def add(self, session: _Session) -> None:
        with self._guard:
            self._sessions[session.id] = session
            while len(self._sessions) > self._max_sessions:
                _, oldest = self._sessions.popitem(last=False)
                oldest.closed = True

    def get(self, session_id: str) -> _Session | None:
        with self._guard:
            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions.move_to_end(session_id)
            return session

    def discard(self, session_id: str) -> None:
        with self._guard:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                session.closed = True


# --- The app ---------------------------------------------------------------------------------


def create_app(
    source: EventSource,
    db_path: Path = DEFAULT_DB_PATH,
    config: AgentConfig = AgentConfig(),
    client: Any = None,
    chat_model: Any = None,
    tracing: bool = False,
    max_sessions: int = MAX_SESSIONS,
) -> FastAPI:
    """Build the app. `client` and `chat_model` let tests pass in fakes instead of real ones."""
    store = ShortlistStore(db_path)
    runs = RunLog(db_path)
    sessions = _SessionStore(max_sessions)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if tracing:
            flush_traces()

    app = FastAPI(title="Event Scout", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    def engine_problem(engine: str) -> str | None:
        """Why this engine can't run right now, or None if it can."""
        if client is not None or chat_model is not None:
            return None
        if engine == "langgraph" and importlib.util.find_spec("langgraph") is None:
            return LANGCHAIN_MISSING_TEXT
        if not has_credentials_for(engine, os.environ):
            return NO_CREDENTIALS_TEXT
        return None

    def start_session(engine: EngineName) -> _Session:
        problem = engine_problem(engine)
        if problem:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, problem)
        recorder = _StepRecorder()
        agent = build_agent(
            engine,
            build_tools(source, store),
            build_system_prompt(date.today(), source.describe()),
            config,
            recorder,
            client=client,
            chat_model=chat_model,
            tracing=tracing,
        )
        session = _Session(uuid.uuid4().hex, engine, agent, recorder)
        sessions.add(session)
        return session

    def find_session(session_id: str, engine: EngineName | None) -> _Session:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, _CONVERSATION_GONE)
        if engine is not None and engine != session.engine:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"This conversation uses the {session.engine} engine. Start a new one to change engines.",
            )
        return session

    def record_run(session: _Session, stop_reason: str, seconds: float, steps: int = 0, usage: TokenUsage | None = None) -> None:
        """Save the measurements. A failure here is logged: the user still gets their answer."""
        usage = usage or TokenUsage()
        try:
            runs.record(
                QuestionRun(
                    session_id=session.id,
                    engine=session.engine,
                    model=config.model,
                    stop_reason=stop_reason,
                    steps=steps,
                    tool_calls=len(session.recorder.steps),
                    usage=usage,
                    cost_usd=None if stop_reason == ERROR else usage.cost(PRICING.get(config.model)),
                    seconds=seconds,
                )
            )
        except (sqlite3.Error, OSError):
            logger.exception("Couldn't record the question run.")

    def ask(session: _Session, message: str) -> ChatResponse:
        session.recorder.steps = []
        started = time.perf_counter()
        try:
            reply = session.agent.ask(message)
        except anthropic.APIError as err:
            # The history may now be incomplete, and the API could reject every later
            # request in this conversation, so it's discarded.
            sessions.discard(session.id)
            record_run(session, ERROR, time.perf_counter() - started)
            raise HTTPException(_status_for(err), describe_api_error(err, config.model)) from None
        except Exception:
            sessions.discard(session.id)
            record_run(session, ERROR, time.perf_counter() - started)
            logger.exception("The agent failed while answering a question.")
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, _FAILED) from None

        seconds = time.perf_counter() - started
        record_run(session, reply.stop_reason, seconds, reply.steps, reply.usage)
        return ChatResponse(
            session_id=session.id,
            engine=session.engine,
            answer=reply.text,
            stop_reason=reply.stop_reason,
            steps=reply.steps,
            tool_calls=session.recorder.steps,
            usage=Usage(
                input_tokens=reply.usage.input_tokens,
                cache_read_tokens=reply.usage.cache_read_tokens,
                cache_write_tokens=reply.usage.cache_write_tokens,
                output_tokens=reply.usage.output_tokens,
            ),
            cost_usd=reply.usage.cost(PRICING.get(config.model)),
            seconds=seconds,
        )

    def open_session(request: ChatRequest) -> _Session:
        if request.session_id is None:
            return start_session(request.engine or "custom")
        return find_session(request.session_id, request.engine)

    def ask_in_turn(session: _Session, message: str, publish: Publish | None = None) -> ChatResponse:
        with session.lock:
            if session.closed:
                raise HTTPException(status.HTTP_404_NOT_FOUND, _CONVERSATION_GONE)
            session.recorder.publish = publish
            try:
                return ask(session, message)
            finally:
                session.recorder.publish = None

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> ChatResponse:
        """Ask a question and wait for the answer. A failed question ends its conversation."""
        return ask_in_turn(open_session(request), request.message)

    @app.post("/api/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        """Ask a question and get progress as it happens.

        Events: model_call, tool_start, and tool_result, then either done (the same fields as
        POST /api/chat) or error (`status` and `detail`). A failed question ends its conversation.
        """
        # Done before the stream starts, so these problems still get a normal error status.
        session = open_session(request)
        events: queue.Queue[str | None] = queue.Queue()

        def publish(event: str, payload: dict[str, Any]) -> None:
            events.put(_server_sent_event(event, payload))

        def run_agent() -> None:
            try:
                publish("done", ask_in_turn(session, request.message, publish).model_dump())
            except HTTPException as err:
                publish("error", {"status": err.status_code, "detail": err.detail})
            except Exception:
                logger.exception("The stream failed while answering a question.")
                publish("error", {"status": status.HTTP_500_INTERNAL_SERVER_ERROR, "detail": _FAILED})
            finally:
                events.put(None)  # tells stream() that nothing more is coming

        def stream() -> Iterator[str]:
            while True:
                try:
                    event = events.get(timeout=KEEP_ALIVE_SECONDS)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                if event is None:
                    return
                yield event

        threading.Thread(target=run_agent, daemon=True).start()
        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            # Asks anything in between not to store or compress the stream, which would hold
            # updates back until the end.
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    @app.delete("/api/chat/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    def end_chat(session_id: str) -> None:
        """End a conversation. Saved events are kept. Ending one that's already gone is fine."""
        sessions.discard(session_id)

    @app.get("/api/shortlist")
    def get_shortlist() -> ShortlistResponse:
        return ShortlistResponse(entries=[ShortlistEntry(**entry) for entry in store.all()])

    @app.delete("/api/shortlist/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
    def remove_from_shortlist(event_id: str) -> None:
        if not store.remove(event_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"'{event_id}' isn't on the shortlist.")

    @app.get("/api/stats")
    def get_stats() -> StatsResponse:
        return StatsResponse(engines=[EngineStats(**row) for row in runs.stats()])

    @app.get("/api/health")
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            data_source=source.name,
            model=config.model,
            available_engines=[engine for engine in ENGINES if engine_problem(engine) is None],
            tracing=tracing,
        )

    @app.exception_handler(ShortlistError)
    @app.exception_handler(sqlite3.Error)
    def database_error(_: Request, err: Exception) -> Response:
        # The full message, which includes the file's location, goes only to the server log.
        logger.error("Database error: %s", err)
        return JSONResponse({"detail": "The database can't be used."}, status.HTTP_500_INTERNAL_SERVER_ERROR)

    return app


_CONVERSATION_GONE = "That conversation has ended."
_FAILED = "Something went wrong answering that."


def _server_sent_event(event: str, payload: dict[str, Any]) -> str:
    # json.dumps writes everything on one line, as this format requires.
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def _status_for(err: anthropic.APIError) -> int:
    if isinstance(err, anthropic.RateLimitError):
        return status.HTTP_429_TOO_MANY_REQUESTS
    return status.HTTP_502_BAD_GATEWAY  # the problem is with a service this server depends on
