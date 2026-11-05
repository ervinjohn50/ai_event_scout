"""Tests for the web API. They run offline, with scripted Claude responses."""

import json
import sqlite3
import sys
import threading
import time
from datetime import date

import anthropic
import httpx2
import pytest

pytest.importorskip("fastapi", reason="needs the web extra")

from fastapi.testclient import TestClient  # noqa: E402

from event_scout import api as api_module  # noqa: E402
from event_scout import server  # noqa: E402
from event_scout.api import MAX_MESSAGE_LENGTH, create_app  # noqa: E402
from event_scout.events import SampleEventSource  # noqa: E402
from event_scout.runs import QuestionRun, RunLog  # noqa: E402
from event_scout.shortlist import ShortlistStore  # noqa: E402
from event_scout.usage import TokenUsage  # noqa: E402
from fakes import FakeClient, response, text, thinking, tool_use  # noqa: E402

TODAY = date(2026, 11, 3)
ENGINES = ["custom", "langgraph"]
REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "scout.db"


def fake_for(engine, responses):
    """The keyword argument that gives create_app a scripted Claude for this engine."""
    if engine == "custom":
        return {"client": FakeClient(responses)}
    pytest.importorskip("langchain", reason="needs the langchain extra")
    from fakes_langchain import scripted_model

    return {"chat_model": scripted_model(responses)}


def make_web_client(db_path, **kwargs):
    source = SampleEventSource(today=TODAY)
    app = create_app(source, db_path, **kwargs)
    return TestClient(app, base_url="http://127.0.0.1")


def chat(web, message="hi", **fields):
    return web.post("/api/chat", json={"message": message, **fields})


# --- Asking questions ------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_a_direct_answer(db_path, engine):
    web = make_web_client(db_path, **fake_for(engine, [response(thinking("Easy."), text("Hello!"))]))
    result = chat(web, engine=engine)

    assert result.status_code == 200
    body = result.json()
    assert body["answer"] == "Hello!"
    assert body["engine"] == engine
    assert body["stop_reason"] == "end_turn"
    assert body["steps"] == 1
    assert body["tool_calls"] == []
    assert body["usage"]["output_tokens"] == 100
    assert body["cost_usd"] > 0
    assert body["session_id"]


@pytest.mark.parametrize("engine", ENGINES)
def test_tool_calls_are_reported_and_saves_reach_the_shortlist(db_path, engine):
    script = [
        response(tool_use("t1", "save_to_shortlist", {"event_id": "sample-002", "note": "date night"}), stop_reason="tool_use"),
        response(tool_use("t2", "get_event_details", {"event_id": "made-up"}), stop_reason="tool_use"),
        response(text("Saved it.")),
    ]
    web = make_web_client(db_path, **fake_for(engine, script))
    body = chat(web, "save it", engine=engine).json()

    assert body["steps"] == 3
    assert body["tool_calls"] == [
        {"tool": "save_to_shortlist", "input": {"event_id": "sample-002", "note": "date night"}, "is_error": False},
        {"tool": "get_event_details", "input": {"event_id": "made-up"}, "is_error": True},
    ]
    entries = web.get("/api/shortlist").json()["entries"]
    assert [(e["event_id"], e["note"]) for e in entries] == [("sample-002", "date night")]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_follow_up_continues_the_same_conversation(db_path, engine):
    fake = fake_for(engine, [response(text("First answer.")), response(text("Second answer."))])
    web = make_web_client(db_path, **fake)
    session_id = chat(web, "first", engine=engine).json()["session_id"]
    body = chat(web, "second", session_id=session_id).json()

    assert body["session_id"] == session_id
    assert body["engine"] == engine
    assert body["tool_calls"] == []
    sent = (fake.get("client") or fake["chat_model"]).requests[-1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]


def test_tool_calls_only_cover_the_latest_question(db_path):
    script = [
        response(tool_use("t1", "list_shortlist", {}), stop_reason="tool_use"),
        response(text("Empty.")),
        response(text("Bye.")),
    ]
    web = make_web_client(db_path, client=FakeClient(script))
    first = chat(web).json()
    second = chat(web, session_id=first["session_id"]).json()
    assert len(first["tool_calls"]) == 1
    assert second["tool_calls"] == []


def test_separate_conversations_have_separate_histories(db_path):
    client = FakeClient([response(text("One.")), response(text("Two."))])
    web = make_web_client(db_path, client=client)
    first = chat(web, "one").json()["session_id"]
    second = chat(web, "two").json()["session_id"]
    assert first != second
    assert client.requests[-1]["messages"] == [{"role": "user", "content": "two"}]


def test_a_cut_off_reply_is_returned_with_its_stop_reason(db_path):
    web = make_web_client(db_path, client=FakeClient([response(text("Partial"), stop_reason="max_tokens")]))
    body = chat(web).json()
    assert body["stop_reason"] == "max_tokens"
    assert "cut off" in body["answer"]


# --- Invalid requests ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": "x" * (MAX_MESSAGE_LENGTH + 1)},
        {"message": "hi", "engine": "made-up"},
    ],
)
def test_invalid_requests_never_reach_the_model(db_path, payload):
    client = FakeClient([])
    web = make_web_client(db_path, client=client)
    assert web.post("/api/chat", json=payload).status_code == 422
    assert client.requests == []


def test_an_unknown_conversation_is_not_found(db_path):
    web = make_web_client(db_path, client=FakeClient([]))
    assert chat(web, session_id="made-up").status_code == 404


def test_a_conversation_cant_change_engines(db_path):
    web = make_web_client(db_path, client=FakeClient([response(text("Hi."))]))
    session_id = chat(web, engine="custom").json()["session_id"]
    result = chat(web, session_id=session_id, engine="langgraph")
    assert result.status_code == 400
    assert "custom" in result.json()["detail"]


def test_ending_a_conversation(db_path):
    web = make_web_client(db_path, client=FakeClient([response(text("Hi."))]))
    session_id = chat(web).json()["session_id"]
    assert web.delete(f"/api/chat/{session_id}").status_code == 204
    assert web.delete(f"/api/chat/{session_id}").status_code == 204
    assert chat(web, session_id=session_id).status_code == 404


# --- Requests from other websites ------------------------------------------------------------


@pytest.mark.parametrize("headers", [{"Content-Type": "text/plain"}, {}], ids=["text/plain", "no content type"])
def test_a_request_that_isnt_marked_as_json_is_refused(db_path, headers):
    """Browsers let any website send these POSTs without asking this server first, while a
    JSON POST needs permission this server never gives. If these were accepted, a page you
    visited could spend your API credit."""
    client = FakeClient([])
    web = make_web_client(db_path, client=client)
    result = web.post("/api/chat", content=json.dumps({"message": "hi"}), headers=headers)
    assert result.status_code == 422
    assert client.requests == []


def test_a_request_for_another_host_name_is_refused(db_path):
    web = make_web_client(db_path, client=FakeClient([]))
    assert web.get("/api/health", headers={"Host": "evil.example.com"}).status_code == 400


def test_other_websites_get_no_cors_permission(db_path):
    web = make_web_client(db_path, client=FakeClient([]))
    result = web.get("/api/shortlist", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in result.headers


# --- Failures --------------------------------------------------------------------------------


class FailingClient(FakeClient):
    def __init__(self, error):
        super().__init__([])
        self.error = error

    def _create(self, **request):
        raise self.error


API_ERRORS = [
    (anthropic.AuthenticationError("bad", response=httpx2.Response(401, request=REQUEST), body=None), 502, "API key was rejected"),
    (anthropic.PermissionDeniedError("no", response=httpx2.Response(403, request=REQUEST), body=None), 502, "doesn't have access"),
    (anthropic.NotFoundError("no", response=httpx2.Response(404, request=REQUEST), body=None), 502, "wasn't found"),
    (anthropic.RateLimitError("slow", response=httpx2.Response(429, request=REQUEST), body=None), 429, "Rate limited"),
    (anthropic.BadRequestError("bad input", response=httpx2.Response(400, request=REQUEST), body=None), 502, "bad input"),
    (anthropic.InternalServerError("oops", response=httpx2.Response(500, request=REQUEST), body=None), 502, "HTTP 500"),
    (anthropic.APIConnectionError(request=REQUEST), 502, "Couldn't reach"),
]


@pytest.mark.parametrize("error, expected_status, expected_text", API_ERRORS)
def test_api_errors_become_readable_responses(db_path, error, expected_status, expected_text):
    web = make_web_client(db_path, client=FailingClient(error))
    result = chat(web)
    assert result.status_code == expected_status
    assert expected_text in result.json()["detail"]


def test_a_failure_ends_the_conversation_and_is_counted(db_path):
    client = FakeClient([response(text("Hi."))])
    web = make_web_client(db_path, client=client)
    session_id = chat(web).json()["session_id"]

    error = anthropic.RateLimitError("slow", response=httpx2.Response(429, request=REQUEST), body=None)
    client.beta.messages.create = FailingClient(error)._create
    assert chat(web, session_id=session_id).status_code == 429
    assert chat(web, session_id=session_id).status_code == 404

    stats = web.get("/api/stats").json()["engines"][0]
    assert stats["questions"] == 2
    assert stats["errors"] == 1
    assert stats["avg_steps"] == 1  # the failed question doesn't pull this down


def test_langgraph_api_errors_are_handled_the_same_way(db_path, monkeypatch):
    fake = fake_for("langgraph", [])
    from fakes_langchain import ScriptedChatAnthropic

    def fail(self, payload):
        raise anthropic.RateLimitError("slow", response=httpx2.Response(429, request=REQUEST), body=None)

    monkeypatch.setattr(ScriptedChatAnthropic, "_create", fail)
    web = make_web_client(db_path, **fake)
    result = chat(web, engine="langgraph")
    assert result.status_code == 429
    assert "Rate limited" in result.json()["detail"]


def test_an_unexpected_failure_doesnt_reveal_details(db_path):
    web = make_web_client(db_path, client=FailingClient(ValueError("secret detail")))
    result = chat(web)
    assert result.status_code == 500
    assert "secret detail" not in result.text


def test_no_credentials_is_explained(db_path, monkeypatch):
    monkeypatch.setattr(api_module, "has_credentials_for", lambda engine, env: False)
    web = make_web_client(db_path)
    result = chat(web)
    assert result.status_code == 503
    assert "No Claude API key" in result.json()["detail"]


def test_missing_langchain_is_explained(db_path, monkeypatch):
    monkeypatch.setattr(api_module, "has_credentials_for", lambda engine, env: True)
    monkeypatch.setattr(api_module.importlib.util, "find_spec", lambda name: None)
    web = make_web_client(db_path)
    result = chat(web, engine="langgraph")
    assert result.status_code == 503
    assert "uv sync --extra langchain" in result.json()["detail"]
    assert web.get("/api/health").json()["available_engines"] == ["custom"]


def test_a_stats_write_failure_doesnt_lose_the_answer(db_path, monkeypatch):
    web = make_web_client(db_path, client=FakeClient([response(text("Hello!"))]))

    def broken(self, run):
        raise sqlite3.OperationalError("disk is full")

    monkeypatch.setattr(RunLog, "record", broken)
    result = chat(web)
    assert result.status_code == 200
    assert result.json()["answer"] == "Hello!"


def test_a_corrupt_database_is_reported_without_its_location(db_path):
    db_path.write_text("this is not a database file, " * 100)
    web = make_web_client(db_path, client=FakeClient([]))
    for path in ("/api/shortlist", "/api/stats"):
        result = web.get(path)
        assert result.status_code == 500
        assert result.json() == {"detail": "The database can't be used."}


# --- Conversations in memory -----------------------------------------------------------------


def test_the_oldest_conversation_is_dropped_at_the_limit(db_path):
    client = FakeClient([response(text("Hi.")) for _ in range(5)])
    web = make_web_client(db_path, client=client, max_sessions=2)
    first = chat(web).json()["session_id"]
    second = chat(web).json()["session_id"]

    chat(web, session_id=first)  # now the second conversation is the least recently used
    chat(web)  # a third conversation pushes it out

    assert chat(web, session_id=second).status_code == 404
    assert chat(web, session_id=first).status_code == 200


def test_simultaneous_questions_in_one_conversation_take_turns(db_path):
    """Without the per-conversation lock, these overlap and the history's order breaks."""

    class SlowClient(FakeClient):
        running = 0
        most_at_once = 0

        def _create(self, **request):
            SlowClient.running += 1
            SlowClient.most_at_once = max(SlowClient.most_at_once, SlowClient.running)
            time.sleep(0.05)
            SlowClient.running -= 1
            return super()._create(**request)

    client = SlowClient([response(text("Answer.")) for _ in range(5)])
    web = make_web_client(db_path, client=client)
    session_id = chat(web).json()["session_id"]

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(chat(web, session_id=session_id).status_code))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [200, 200, 200, 200]
    assert SlowClient.most_at_once == 1
    roles = [m["role"] for m in client.requests[-1]["messages"]]
    assert roles == ["user", "assistant"] * 4 + ["user"]


def test_a_question_waiting_on_a_failed_conversation_is_told_it_ended(db_path):
    """The first question fails while the second is waiting its turn. The second must not
    continue on a history that's been discarded."""
    in_flight = threading.Event()
    release = threading.Event()

    class FailsOnceWaiting(FakeClient):
        def _create(self, **request):
            if len(self.requests) == 1:  # the opening question already succeeded
                self.requests.append(request)
                in_flight.set()
                release.wait(timeout=10)
                raise anthropic.APIConnectionError(request=REQUEST)
            return super()._create(**request)

    client = FailsOnceWaiting([response(text("Hi."))])
    web = make_web_client(db_path, client=client)
    session_id = chat(web).json()["session_id"]

    results = {}
    first = threading.Thread(target=lambda: results.update(first=chat(web, session_id=session_id).status_code))
    second = threading.Thread(target=lambda: results.update(second=chat(web, session_id=session_id).status_code))
    first.start()
    assert in_flight.wait(timeout=10)
    second.start()
    time.sleep(0.1)  # gives the second request time to reach the lock
    release.set()
    first.join()
    second.join()

    assert results == {"first": 502, "second": 404}
    assert len(client.requests) == 2  # the second question never reached the model


# --- Shortlist -------------------------------------------------------------------------------


def test_shortlist_starts_empty_and_needs_no_credentials(db_path, monkeypatch):
    monkeypatch.setattr(api_module, "has_credentials_for", lambda engine, env: False)
    web = make_web_client(db_path)
    assert web.get("/api/shortlist").json() == {"entries": []}


def test_removing_from_the_shortlist(db_path):
    source = SampleEventSource(today=TODAY)
    ShortlistStore(db_path).add(source.get("sample-002"), "date night")
    web = make_web_client(db_path, client=FakeClient([]))

    entry = web.get("/api/shortlist").json()["entries"][0]
    assert entry["name"] == "Lake Effect Brass Band"
    assert entry["note"] == "date night"

    assert web.delete("/api/shortlist/sample-002").status_code == 204
    assert web.delete("/api/shortlist/sample-002").status_code == 404
    assert web.get("/api/shortlist").json() == {"entries": []}


# --- Stats -----------------------------------------------------------------------------------


def run(engine="custom", stop_reason="end_turn", steps=2, tool_calls=1, cost=0.01, seconds=1.0, input_tokens=100):
    usage = TokenUsage(input_tokens=input_tokens, cache_read_tokens=50, cache_write_tokens=50, output_tokens=20)
    return QuestionRun("session", engine, "claude-sonnet-5-5", stop_reason, steps, tool_calls, usage, cost, seconds)


def test_stats_are_grouped_by_engine(db_path):
    log = RunLog(db_path)
    log.record(run("custom", steps=2, tool_calls=1, cost=0.01, seconds=1.0, input_tokens=100))
    log.record(run("custom", steps=4, tool_calls=3, cost=0.03, seconds=3.0, input_tokens=300))
    log.record(run("custom", "error", steps=0, tool_calls=0, cost=None, seconds=9.0, input_tokens=0))
    log.record(run("langgraph", steps=1, tool_calls=0, cost=0.02, seconds=2.0, input_tokens=100))

    web = make_web_client(db_path, client=FakeClient([]))
    custom, langgraph = web.get("/api/stats").json()["engines"]

    assert custom["engine"] == "custom"
    assert custom["questions"] == 3
    assert custom["errors"] == 1
    assert custom["avg_steps"] == 3
    assert custom["avg_tool_calls"] == 2
    assert custom["avg_seconds"] == 2
    assert custom["avg_prompt_tokens"] == 300
    assert custom["avg_output_tokens"] == 20
    assert custom["total_cost_usd"] == pytest.approx(0.04)
    assert langgraph["engine"] == "langgraph"
    assert langgraph["questions"] == 1


def test_stats_are_empty_before_any_questions(db_path):
    web = make_web_client(db_path, client=FakeClient([]))
    assert web.get("/api/stats").json() == {"engines": []}


def test_an_engine_with_only_failures_has_no_averages(db_path):
    RunLog(db_path).record(run("custom", "error", steps=0, tool_calls=0, cost=None))
    web = make_web_client(db_path, client=FakeClient([]))
    stats = web.get("/api/stats").json()["engines"][0]
    assert stats["errors"] == 1
    assert stats["avg_steps"] is None
    assert stats["total_cost_usd"] == 0


def test_run_records_dont_keep_the_question_or_answer(db_path):
    web = make_web_client(db_path, client=FakeClient([response(text("secret answer"))]))
    chat(web, "secret question")
    connection = sqlite3.connect(db_path)
    rows = connection.execute("SELECT * FROM question_runs").fetchall()
    connection.close()
    assert len(rows) == 1
    assert "secret" not in repr(rows)


# --- Health ----------------------------------------------------------------------------------


def test_health(db_path):
    web = make_web_client(db_path, client=FakeClient([]))
    assert web.get("/api/health").json() == {
        "status": "ok",
        "data_source": "sample",
        "model": "claude-sonnet-5-5",
        "available_engines": ["custom", "langgraph"],
        "tracing": False,
    }


# --- Starting the server ---------------------------------------------------------------------


def test_the_server_only_listens_on_this_computer(db_path, monkeypatch):
    import uvicorn

    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: started.update(options))
    monkeypatch.setattr(server, "load_dotenv", lambda path: None)
    assert server.main(["--offline", "--db-path", str(db_path), "--port", "9000"]) == 0
    assert started == {"host": "127.0.0.1", "port": 9000}


def test_missing_web_packages_print_an_install_hint(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "uvicorn", None)  # makes `import uvicorn` raise ImportError
    monkeypatch.setattr(server, "load_dotenv", lambda path: None)
    assert server.main([]) == 2
    assert "uv sync --extra web" in capsys.readouterr().err
