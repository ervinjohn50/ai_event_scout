"""Tests for the terminal CLI, driven with a fake Claude and scripted keyboard input."""

import json
import os

import anthropic
import httpx2
import pytest
from fakes import FakeClient, response, text, thinking, tool_use

from event_scout import cli
from event_scout.events import SampleEventSource, TicketmasterEventSource
from event_scout.shortlist import ShortlistStore


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Run every test in an empty folder with an empty home folder and no API keys,
    so a real .env file or local credentials can't affect the results."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "TICKETMASTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def type_lines(monkeypatch, *lines):
    """Pretend the user types these lines at the prompt, then presses Ctrl-D."""
    feed = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)


def answer(reply_text="Here you go."):
    return [response(thinking("Simple."), text(reply_text))]


# --- Setup helpers -----------------------------------------------------------------------


def test_load_dotenv_reads_values_but_never_overrides_the_shell(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# a comment\n"
        "ANTHROPIC_API_KEY='sk-from-file'\n"
        "export TICKETMASTER_API_KEY=tm-from-file\n"
        "EMPTY=\n"
        "not a setting\n"
    )
    monkeypatch.setenv("TICKETMASTER_API_KEY", "tm-from-shell")

    cli.load_dotenv(tmp_path / ".env")

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-from-file"  # quotes stripped
    assert os.environ["TICKETMASTER_API_KEY"] == "tm-from-shell"  # shell wins
    assert "EMPTY" not in os.environ
    monkeypatch.delenv("ANTHROPIC_API_KEY")


def test_load_dotenv_without_a_file_does_nothing(tmp_path):
    cli.load_dotenv(tmp_path / ".env")


def test_credentials_detection(tmp_path):
    assert not cli.has_anthropic_credentials({})
    assert cli.has_anthropic_credentials({"ANTHROPIC_API_KEY": "sk-x"})
    assert cli.has_anthropic_credentials({"ANTHROPIC_AUTH_TOKEN": "tok"})
    (tmp_path / ".config" / "anthropic").mkdir(parents=True)  # an `ant auth login` profile
    assert cli.has_anthropic_credentials({})


def test_choose_source():
    assert isinstance(cli.choose_source(False, {}), SampleEventSource)
    assert isinstance(cli.choose_source(False, {"TICKETMASTER_API_KEY": "k"}), TicketmasterEventSource)
    assert isinstance(cli.choose_source(True, {"TICKETMASTER_API_KEY": "k"}), SampleEventSource)


def test_missing_api_key_exits_with_instructions(capsys):
    assert cli.main(["hello"]) == 2
    err = capsys.readouterr().err
    assert "No Claude API key found" in err and ".env.example" in err


# --- One-shot questions --------------------------------------------------------------------


def test_one_shot_question_prints_trace_and_answer(capsys):
    client = FakeClient(
        [
            response(tool_use("t1", "search_events", {"city": "Austin"}), stop_reason="tool_use"),
            response(text("Gravel Hymnal is playing Thursday.")),
        ]
    )

    assert cli.main(["What's on in Austin?"], client=client) == 0

    out = capsys.readouterr().out
    assert 'tool call  search_events {"city": "Austin"}' in out
    assert "done in 2 steps" in out
    assert out.rstrip().endswith("Gravel Hymnal is playing Thursday.")


def test_quiet_mode_prints_only_the_answer(capsys):
    assert cli.main(["--quiet", "hi"], client=FakeClient(answer("Hello!"))) == 0
    assert capsys.readouterr().out.strip() == "Hello!"


def test_flags_change_the_request():
    client = FakeClient(answer())
    cli.main(["--quiet", "--model", "claude-opus-5-5", "--effort", "low", "--no-fallbacks", "hi"], client=client)

    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"  # not the default, so this proves the flag works
    assert request["output_config"] == {"effort": "low"}
    assert "fallbacks" not in request
    assert "OFFLINE SAMPLE" in request["system"]


class ErrorClient:
    """A client whose every request fails with the given SDK exception."""

    def __init__(self, error):
        self.beta = self
        self.messages = self
        self.error = error

    def create(self, **request):
        raise self.error


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=REQUEST), body=None), "API key was rejected"),
        (anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=REQUEST), body=None), "Rate limited"),
        (anthropic.NotFoundError("no model", response=httpx2.Response(404, request=REQUEST), body=None), "wasn't found"),
        (anthropic.InternalServerError("oops", response=httpx2.Response(500, request=REQUEST), body=None), "HTTP 500"),
        (anthropic.APIConnectionError(request=REQUEST), "Couldn't reach the Claude API"),
    ],
)
def test_api_errors_become_readable_messages(capsys, error, expected):
    assert cli.main(["--quiet", "hi"], client=ErrorClient(error)) == 1
    assert expected in capsys.readouterr().err


# --- Interactive chat ----------------------------------------------------------------------


def test_chat_session_with_commands(monkeypatch, capsys):
    client = FakeClient(answer("Plenty of shows!"))
    type_lines(monkeypatch, "/help", "/shortlist", "", "anything this week?", "/history", "/reset", "/bogus", "/quit")

    assert cli.main(["--quiet"], client=client) == 0

    out = capsys.readouterr().out
    assert "Event Scout | claude-sonnet-5-5, effort medium | offline sample data" in out
    assert "/history" in out  # from /help
    assert "Your shortlist is empty" in out
    assert "Plenty of shows!" in out
    assert '"role": "user"' in out and '"signature": "<14 chars>"' in out  # /history output
    assert "Started a new conversation." in out
    assert "Unknown command /bogus" in out
    assert len(client.requests) == 1  # commands never call the API


def test_ctrl_d_at_the_prompt_exits_cleanly(monkeypatch):
    type_lines(monkeypatch)  # no lines: the first input() raises EOFError
    assert cli.main([], client=FakeClient([])) == 0


def test_ctrl_c_during_an_answer_resets_the_conversation(monkeypatch, capsys):
    class InterruptedClient(ErrorClient):
        def __init__(self):
            super().__init__(KeyboardInterrupt())

    type_lines(monkeypatch, "find shows")
    assert cli.main(["--quiet"], client=InterruptedClient()) == 0
    assert "Interrupted. Started a new conversation." in capsys.readouterr().out


def test_print_shortlist_formats_entries(tmp_path, capsys):
    store = ShortlistStore(tmp_path / "s.db")
    store.add(SampleEventSource().get("sample-002"), note="date night")

    cli.print_shortlist(store)

    line = capsys.readouterr().out.strip()
    assert "Lake Effect Brass Band  @ Green Door Jazz Club, Chicago  (20-35 USD)  note: date night" in line


def test_history_command_prints_valid_json(monkeypatch, capsys):
    client = FakeClient(answer())
    type_lines(monkeypatch, "hi", "/history")
    cli.main(["--quiet"], client=client)

    out = capsys.readouterr().out
    history = json.loads(out[out.index("[") :])
    assert [m["role"] for m in history] == ["user", "assistant"]
