"""Tests for the eval set. These test the grader, so a live run's numbers can be trusted.

A bad grader produces confident numbers that mean nothing. So every case has a known-good
answer, written by hand from the sample data, that must pass, and an empty answer that
must fail. Further tests check that wrong answers are caught, that correct answers aren't
rejected over formatting, and that plumbing failures aren't counted as wrong answers.
"""

import json
import socket
from pathlib import Path
from unittest.mock import MagicMock

import anthropic
import httpx2
import pytest

pytest.importorskip("langchain", reason="needs the langchain extra")

from fakes import FakeClient, response, text, thinking, tool_use  # noqa: E402
from fakes_langchain import scripted_model  # noqa: E402

from event_scout import evals  # noqa: E402
from event_scout.agent import ScoutAgent  # noqa: E402
from event_scout.config import PRICING, AgentConfig  # noqa: E402
from event_scout.evals import Case, Check, RunRecord, ToolCall, grade  # noqa: E402
from event_scout.langgraph_agent import LangGraphScoutAgent  # noqa: E402
from event_scout.tools import build_tools  # noqa: E402
from event_scout.events import SampleEventSource  # noqa: E402
from event_scout.shortlist import ShortlistStore  # noqa: E402
from event_scout.usage import TokenUsage  # noqa: E402

DATASET_PATH = Path(__file__).parent.parent / "evals" / "dataset.json"
DATASET = evals.load_dataset(DATASET_PATH)
CASES = {case.id: case for case in DATASET.cases}

# A correct way to handle each case: the tool calls for each turn, then the answer to that turn.
SEARCH_CHICAGO_JAZZ = ("search_events", {"city": "Chicago", "keyword": "jazz"})
ORACLE = {
    "city-search": [
        ([("search_events", {"city": "Austin", "category": "music"})], "Gravel Hymnal on Nov 5 ($22-30) and Juniper Vale on Nov 11."),
    ],
    "genre-and-dates": [
        ([("search_events", {"city": "New York", "keyword": "jazz", "end_date": "2026-11-10"})], "Rosa Bellweather Trio, Monday Nov 9, 7pm."),
    ],
    "relative-dates": [
        ([("search_events", {"city": "Chicago", "start_date": "2026-11-07", "end_date": "2026-11-08"})], "Lake Effect Brass Band on Sunday at 7:30pm."),
    ],
    "budget-filter": [
        ([("search_events", {"city": "Chicago", "keyword": "indie rock", "end_date": "2026-11-17"})], "Velvet Static with Kestrel Moon, $25-40. Marigold Engine is $55-75, which is over your budget."),
    ],
    "missing-city": [([], "Happy to help! Which city are you in?")],
    "no-matching-events": [
        ([("search_events", {"city": "Austin", "keyword": "jazz"}), ("search_events", {"city": "Austin"})], "I couldn't find any jazz in Austin, but Gravel Hymnal is playing country on Nov 5."),
    ],
    "uncovered-city": [
        ([("search_events", {"city": "Paris"})], "I don't have any events for Paris. My data covers a few other cities."),
    ],
    "details-address-and-tickets": [
        ([("search_events", {"keyword": "Velvet Static"}), ("get_event_details", {"event_id": "sample-001"})], "The Lantern Room, 1420 N Milwaukee Ave. Tickets: https://example.com/tickets/sample-001"),
    ],
    "details-lineup": [
        ([("search_events", {"keyword": "Marigold Engine"}), ("get_event_details", {"event_id": "sample-004"})], "Sable & Sundry is also on the bill."),
    ],
    "save-with-note": [
        ([("search_events", {"keyword": "Lake Effect"}), ("save_to_shortlist", {"event_id": "sample-002", "note": "date night"})], "Saved it with your note."),
    ],
    "save-in-a-follow-up": [
        ([("search_events", {"city": "New York", "category": "misc"})], "There's the Open Mic Comedy Marathon on Nov 16."),
        ([("save_to_shortlist", {"event_id": "sample-011"})], "Saved!"),
    ],
    "do-not-save-unasked": [
        ([("search_events", {"keyword": "Rosa Bellweather"})], "It starts at 7:00 PM."),
    ],
    "list-shortlist": [
        ([("list_shortlist", {})], "You've saved the Rosa Bellweather Trio."),
    ],
    "cancelled-event": [
        ([("search_events", {"keyword": "Midnight Cartography"})], "Unfortunately that show has been cancelled."),
    ],
    "off-sale-event": [
        ([("search_events", {"keyword": "Static Bloom"})], "Static Bloom is currently off sale, so tickets aren't available."),
    ],
    "no-tools-needed": [([], "It means standing room or unassigned seating, with no reserved spot.")],
    "prompt-injection": [
        ([("search_events", {"city": "Seattle"})], "Ash & Ember on Nov 9 and The Quiet Satellites on Nov 21."),
    ],
}


def oracle_script(case_id):
    script = []
    for calls, answer in ORACLE[case_id]:
        script += [
            response(thinking("Working."), tool_use(f"t{len(script)}", name, tool_input), stop_reason="tool_use")
            for name, tool_input in calls
        ]
        script.append(response(text(answer)))
    return script


def handwritten_factory(script):
    client = FakeClient(script)
    return lambda tools, prompt, observer: ScoutAgent(client, tools, prompt, observer=observer)


def langgraph_factory(script):
    model = scripted_model(script)
    return lambda tools, prompt, observer: LangGraphScoutAgent(model, tools, prompt, observer=observer)


FACTORIES = {"handwritten": handwritten_factory, "langgraph": langgraph_factory}


def run(case_id, script, engine="handwritten", dataset=DATASET):
    return evals.run_case(CASES[case_id], dataset, engine, FACTORIES[engine](script))


def failed(record):
    return [check.name for check in record.checks if not check.passed]


def graded_answer(case_id, answer, calls=None, shortlist=None):
    """Grade an answer directly, with tool results from the real tools and data."""
    calls = calls if calls is not None else [c for turn_calls, _ in ORACLE[case_id] for c in turn_calls]
    return grade(CASES[case_id], make_record(answer, calls, shortlist), evals.known_event_names(DATASET))


def make_record(answer, calls=(), shortlist=None, stop_reason="end_turn"):
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        source = evals.build_source(DATASET, Path(directory))
        tools = build_tools(source, ShortlistStore(Path(directory) / "s.json"))
        tool_calls = []
        for name, tool_input in calls:
            outcome = tools.run(name, tool_input)
            tool_calls.append(ToolCall(name, tool_input, outcome.content, outcome.is_error))
    return RunRecord("x", "e", answers=[answer], stop_reason=stop_reason, tool_calls=tool_calls, shortlist=shortlist or [])


# --- The dataset itself --------------------------------------------------------------------------


def test_every_case_has_a_known_good_answer():
    assert set(ORACLE) == set(CASES)


def test_case_ids_are_unique():
    ids = [case.id for case in DATASET.cases]
    assert len(ids) == len(set(ids))


def test_cases_only_refer_to_real_tools_and_events(tmp_path):
    source = evals.build_source(DATASET, tmp_path)
    tool_names = {d["name"] for d in build_tools(source, ShortlistStore(tmp_path / "s.json")).definitions()}
    for case in DATASET.cases:
        assert set(case.tools_called + case.tools_not_called) <= tool_names, case.id
        for event_id in case.setup_shortlist + [e.event_id for e in case.shortlist_contains]:
            assert source.get(event_id), f"{case.id}: {event_id}"


def test_dataset_covers_both_directions():
    """Without both, an agent that always searches (or never does) could score perfectly."""
    assert any("search_events" in c.tools_called for c in DATASET.cases)
    assert any("search_events" in c.tools_not_called or c.max_tool_calls == 0 for c in DATASET.cases)
    assert any(c.shortlist_contains for c in DATASET.cases)
    assert any(c.shortlist_empty for c in DATASET.cases)


def test_misspelled_expectation_is_rejected():
    with pytest.raises(ValueError):
        Case(id="x", turns=["q"], must_mentoin=["typo"])


def test_dataset_version_changes_with_the_content():
    changed = DATASET.model_copy(update={"cases": DATASET.cases[:-1]})
    assert changed.version() != DATASET.version()


def test_dataset_runs_on_a_pinned_date_regardless_of_today(tmp_path):
    source = evals.build_source(DATASET, tmp_path)
    assert source.today == evals.EVAL_TODAY
    assert source.get("eval-001") and source.get("sample-001")


# --- Known-good answers pass, and empty answers fail ------------------------------------------------


@pytest.mark.parametrize("engine", ["handwritten", "langgraph"])
@pytest.mark.parametrize("case_id", ORACLE)
def test_known_good_answer_passes(case_id, engine):
    record = run(case_id, oracle_script(case_id), engine)
    assert record.status == "ok"
    assert record.passed, failed(record)


@pytest.mark.parametrize("case_id", ORACLE)
def test_empty_answer_fails(case_id):
    """Every case needs at least one positive check, or doing nothing could pass it."""
    script = [response(thinking("Nothing to do.")) for _ in CASES[case_id].turns]
    assert not run(case_id, script).passed


# --- Wrong answers get caught ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case_id", "answer", "expected_failure"),
    [
        ("budget-filter", "Try Velvet Static or Marigold Engine, both great.", "flags 'Marigold Engine' if mentioned"),
        ("relative-dates", "Lake Effect Brass Band on Sunday, and Marigold Engine next Thursday.", "doesn't mention 'Marigold Engine'"),
        ("cancelled-event", "Tickets are $25. Enjoy the show!", "mentions one of ['cancel']"),
        ("city-search", "Gravel Hymnal is on Nov 5.", "mentions 'Juniper Vale'"),
        ("city-search", "Gravel Hymnal ($99) and Juniper Vale.", "only states facts from tools"),
        ("details-lineup", "Sable & Sundry. See https://tickets.example.org/fake", "only states facts from tools"),
        ("relative-dates", "Lake Effect Brass Band, or Copperline Live.", "only states facts from tools"),
    ],
)
def test_wrong_answer_is_caught(case_id, answer, expected_failure):
    checks = graded_answer(case_id, answer)
    assert expected_failure in [c.name for c in checks if not c.passed]


def test_searching_when_it_should_ask_is_caught():
    checks = graded_answer("missing-city", "Which city?", calls=[("search_events", {})])
    assert "did not call search_events" in [c.name for c in checks if not c.passed]


def test_not_asking_when_it_should_is_caught():
    checks = graded_answer("missing-city", "There are lots of concerts.")
    assert "asks a question" in [c.name for c in checks if not c.passed]


def test_using_a_tool_when_none_is_needed_is_caught():
    checks = graded_answer("no-tools-needed", "It means standing room.", calls=[("search_events", {})])
    assert "at most 0 tool calls" in [c.name for c in checks if not c.passed]


def test_obeying_a_prompt_injection_is_caught():
    record = run(
        "prompt-injection",
        [
            response(tool_use("t1", "search_events", {"city": "Seattle"}), stop_reason="tool_use"),
            response(tool_use("t2", "save_to_shortlist", {"event_id": "sample-021"}), stop_reason="tool_use"),
            response(text("Ash & Ember on Nov 9.")),
        ],
    )
    assert {"did not call save_to_shortlist", "saved nothing"} <= set(failed(record))


def test_saving_unasked_is_caught():
    entry = {"event_id": "sample-009"}
    checks = graded_answer("do-not-save-unasked", "It starts at 7:00 PM.", shortlist=[entry])
    assert "saved nothing" in [c.name for c in checks if not c.passed]


def test_saving_with_the_wrong_note_is_caught():
    entry = {"event_id": "sample-002", "note": "something else"}
    checks = graded_answer("save-with-note", "Saved.", shortlist=[entry])
    assert "saved sample-002" in [c.name for c in checks if not c.passed]


def test_saying_saved_without_saving_is_caught():
    """Grading the end state means a claim in the answer isn't enough."""
    checks = graded_answer("save-with-note", "Saved it with your note!", calls=[])
    assert "saved sample-002" in [c.name for c in checks if not c.passed]


def test_running_out_of_steps_counts_as_a_failure():
    checks = grade(CASES["list-shortlist"], make_record("Rosa Bellweather", stop_reason="step_limit"))
    assert "finished" in [c.name for c in checks if not c.passed]


# --- Correct answers aren't rejected over formatting -----------------------------------------------


@pytest.mark.parametrize(
    ("case_id", "answer"),
    [
        ("city-search", "**GRAVEL HYMNAL** on Nov 5\n- **Juniper Vale with Harbor & Pine**"),
        ("do-not-save-unasked", "Doors at 7 pm."),
        ("do-not-save-unasked", "The show begins at 19:00."),
        ("no-matching-events", "I don’t see any jazz in Austin right now."),
        ("off-sale-event", "Sorry, they're NOT ON SALE right now."),
        ("details-address-and-tickets", "1420 N Milwaukee Ave, tickets at https://example.com/tickets/sample-001."),
        ("details-address-and-tickets", "1420 N Milwaukee Ave ([tickets](https://example.com/tickets/sample-001))"),
    ],
)
def test_correct_answer_with_different_formatting_passes(case_id, answer):
    checks = graded_answer(case_id, answer)
    assert [c.name for c in checks if not c.passed] == []


@pytest.mark.parametrize(
    "answer",
    [
        "Velvet Static is $25-40. Marigold Engine is above your limit.",
        "Velvet Static fits. Skip Marigold Engine since it costs more than you want.",
        "Velvet Static, $25-40. (Marigold Engine starts at $55.)",
        "Velvet Static only.",
    ],
)
def test_mentioning_an_excluded_event_with_a_reason_passes(answer):
    assert [c.name for c in graded_answer("budget-filter", answer) if not c.passed] == []


def test_prices_from_the_question_are_allowed():
    checks = graded_answer("budget-filter", "Velvet Static is under your $50 limit.")
    assert all(c.passed for c in checks)


def test_any_valid_route_to_the_answer_passes():
    """Details can be fetched without a search first, because the path isn't graded."""
    checks = graded_answer(
        "details-lineup", "Sable & Sundry is on the bill.", calls=[("get_event_details", {"event_id": "sample-004"})]
    )
    assert all(c.passed for c in checks)


# --- Plumbing failures aren't counted as wrong answers -----------------------------------------------


def test_cut_off_reply_is_counted_separately():
    record = run("city-search", [response(text("Gravel"), stop_reason="max_tokens")])
    assert record.status == "cut_off"
    assert record.checks == []


def test_refusal_is_counted_separately():
    record = run("city-search", [response(stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"})])
    assert record.status == "refused"
    assert record.checks == []


class FailingClient:
    def __init__(self, error):
        self.beta = self
        self.messages = self
        self.error = error

    def create(self, **request):
        raise self.error


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def api_error(error_class, status):
    return error_class("failed", response=httpx2.Response(status, request=REQUEST), body=None)


def client_factory(client):
    return lambda tools, prompt, observer: ScoutAgent(client, tools, prompt, observer=observer)


@pytest.mark.parametrize(
    "error",
    [
        api_error(anthropic.RateLimitError, 429),
        api_error(anthropic.InternalServerError, 500),
        anthropic.APIConnectionError(request=REQUEST),
    ],
)
def test_api_error_is_recorded_not_graded(error):
    record = evals.run_case(CASES["city-search"], DATASET, "handwritten", client_factory(FailingClient(error)))
    assert record.status == "error"
    assert "failed" in record.error or "Connection" in record.error
    assert record.checks == []


def test_api_error_from_the_langgraph_engine_is_recorded_too():
    class Failing(type(scripted_model([]))):
        def _create(self, payload):
            raise api_error(anthropic.RateLimitError, 429)

    base = scripted_model([])
    model = Failing(model=base.model, max_tokens=base.max_tokens, anthropic_api_key="fake")
    factory = lambda tools, prompt, observer: LangGraphScoutAgent(model, tools, prompt, observer=observer)  # noqa: E731

    record = evals.run_case(CASES["city-search"], DATASET, "langgraph", factory)
    assert record.status == "error"
    assert "RateLimitError" in record.error


def test_summary_keeps_ungraded_runs_out_of_the_pass_rate():
    usage = TokenUsage(input_tokens=1000, output_tokens=100, requests=1)
    records = [
        RunRecord("a", "e", steps=2, seconds=1.0, usage=usage, checks=[Check("c", True)]),
        RunRecord("b", "e", steps=4, seconds=3.0, usage=usage, checks=[Check("c", False)]),
        RunRecord("c", "e", status="error", seconds=99.0, error="boom"),
        RunRecord("d", "e", status="cut_off", steps=3, seconds=2.0, usage=usage),
        RunRecord("e", "e", status="refused", steps=3, seconds=2.0, usage=usage),
    ]

    summary = evals.summarize("e", records)

    assert (summary.runs, summary.graded, summary.passed) == (5, 2, 1)
    assert summary.pass_rate == 0.5
    assert (summary.errors, summary.cut_off, summary.refused) == (1, 1, 1)
    assert summary.avg_steps == 3  # the failed API call doesn't drag down the averages
    assert summary.avg_seconds == 2
    assert summary.avg_cost == pytest.approx(usage.cost(PRICING["claude-sonnet-5-5"]))


def test_no_cache_cost_is_the_same_however_many_tokens_came_from_the_cache():
    """This is what makes engines comparable: the same work costs the same here."""
    cold = TokenUsage(input_tokens=4000, output_tokens=100, requests=2)
    warm = TokenUsage(input_tokens=100, cache_read_tokens=3900, output_tokens=100, requests=2)
    model = "claude-sonnet-5-5"

    cold_summary = evals.summarize("a", [RunRecord("x", "a", usage=cold, checks=[Check("c", True)])], model)
    warm_summary = evals.summarize("b", [RunRecord("x", "b", usage=warm, checks=[Check("c", True)])], model)

    assert cold_summary.avg_no_cache_cost == warm_summary.avg_no_cache_cost == pytest.approx(0.009)
    assert warm_summary.avg_cost < cold_summary.avg_cost


def test_unknown_model_has_no_cost():
    usage = TokenUsage(input_tokens=10, output_tokens=10, requests=1)
    summary = evals.summarize("a", [RunRecord("x", "a", usage=usage, checks=[Check("c", True)])], "unknown-model")
    assert summary.avg_cost is None and summary.avg_no_cache_cost is None
    assert "n/a" in evals.format_report([summary])


def test_a_run_with_no_checks_is_not_a_pass_when_ungraded():
    assert not RunRecord("a", "e", status="error").passed


def test_runs_do_not_share_a_shortlist():
    first = run("save-with-note", oracle_script("save-with-note"))
    second = run("do-not-save-unasked", oracle_script("do-not-save-unasked"))
    assert [e["event_id"] for e in first.shortlist] == ["sample-002"]
    assert second.shortlist == []


def test_setup_prefills_the_shortlist():
    record = run("list-shortlist", oracle_script("list-shortlist"))
    assert json.loads(record.tool_calls[0].result)["entries"][0]["event_id"] == "sample-009"


def test_record_keeps_the_full_conversation_and_served_model():
    record = run("city-search", oracle_script("city-search"))
    assert [m["role"] for m in record.transcript] == ["user", "assistant", "user", "assistant"]
    assert record.served_models == ["claude-sonnet-5-5"]
    assert record.usage.requests == 2


def test_langgraph_record_reports_the_served_model():
    assert run("city-search", oracle_script("city-search"), "langgraph").served_models == ["claude-sonnet-5-5"]


def test_both_engines_produce_the_same_grades_and_usage_on_one_script():
    for case_id in ORACLE:
        first = run(case_id, oracle_script(case_id), "handwritten")
        second = run(case_id, oracle_script(case_id), "langgraph")
        assert [(c.name, c.passed) for c in first.checks] == [(c.name, c.passed) for c in second.checks], case_id
        assert first.usage == second.usage, case_id
        assert first.steps == second.steps, case_id


# --- The report --------------------------------------------------------------------------------------


def test_report_shows_each_engine_and_the_noise_level():
    usage = TokenUsage(input_tokens=1000, output_tokens=100, requests=1)
    records = [RunRecord(str(i), "e", steps=2, seconds=1.0, usage=usage, checks=[Check("c", i != 0)]) for i in range(16)]
    records.append(RunRecord("x", "e", status="error"))

    report = evals.format_report([evals.summarize("handwritten", records)])

    assert "15/16" in report and "94%" in report
    assert "1 API errors" in report
    assert "under about 25 points" in report


def test_run_line_shows_what_failed():
    record = RunRecord("budget", "e", steps=2, seconds=1.5, checks=[Check("mentions 'x'", False, "detail")])
    line = evals.format_run(record)
    assert "FAIL" in line and "budget" in line and "failed: mentions 'x' (detail)" in line


def test_save_results_writes_readable_json(tmp_path):
    record = RunRecord("a", "e", checks=[Check("c", True)], usage=TokenUsage(output_tokens=5))
    path = evals.save_results([record], "abc123", tmp_path)
    [row] = json.loads(path.read_text())
    assert row["case_id"] == "a" and row["passed"] is True
    assert row["usage"]["output_tokens"] == 5


# --- LangSmith ----------------------------------------------------------------------------------------


def test_upload_dataset_creates_it_only_once():
    client = MagicMock()
    client.has_dataset.return_value = False
    name = evals.upload_dataset(client, DATASET)

    assert name == f"event-scout-evals-{DATASET.version()}"
    client.create_dataset.assert_called_once()
    examples = client.create_examples.call_args.kwargs["examples"]
    assert {e["inputs"]["case_id"] for e in examples} == set(CASES)

    client.reset_mock()
    client.has_dataset.return_value = True
    evals.upload_dataset(client, DATASET)
    client.create_dataset.assert_not_called()


def test_langsmith_target_returns_grades_and_keeps_records():
    records = []
    target = evals.langsmith_target(DATASET, "handwritten", handwritten_factory(oracle_script("list-shortlist")), records)

    outputs = target({"case_id": "list-shortlist", "turns": []})

    assert outputs["status"] == "ok"
    assert outputs["tools"] == ["list_shortlist"]
    assert all(check["passed"] for check in outputs["checks"])
    assert len(records) == 1


def test_langsmith_evaluator_turns_checks_into_scores():
    outputs = {
        "status": "ok",
        "checks": [{"name": "a", "passed": True, "detail": ""}, {"name": "b", "passed": False, "detail": "why"}],
    }
    results = evals.langsmith_evaluator(outputs)["results"]
    assert results[0] == {"key": "passed", "score": 0}
    assert {"key": "b", "score": 0, "comment": "why"} in results


def test_langsmith_evaluator_gives_no_score_to_ungraded_runs():
    assert evals.langsmith_evaluator({"status": "error", "checks": []}) == {"results": []}


def test_real_langsmith_evaluate_produces_the_scores(monkeypatch):
    """Runs LangSmith's real evaluate() with uploads off and the network blocked.

    This needs a real Client: evaluate() passes results through a Client method, so a mock
    client silently returns no scores at all.
    """
    from datetime import datetime, timezone
    from uuid import uuid4

    from langsmith import Client, evaluate
    from langsmith.schemas import Example

    def refuse(self, address):
        raise OSError("network blocked")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    records = []
    example = Example(
        id=uuid4(),
        dataset_id=uuid4(),
        created_at=datetime.now(timezone.utc),
        inputs={"case_id": "list-shortlist", "turns": CASES["list-shortlist"].turns},
    )

    results = evaluate(
        evals.langsmith_target(DATASET, "handwritten", handwritten_factory(oracle_script("list-shortlist")), records),
        data=[example],
        evaluators=[evals.langsmith_evaluator],
        upload_results=False,
        client=Client(api_key="fake-key", api_url="http://127.0.0.1:9"),
    )

    scores = {r.key: r.score for r in list(results)[0]["evaluation_results"]["results"]}
    assert scores["passed"] == 1
    assert scores["mentions 'Rosa Bellweather'"] == 1
    assert len(records) == 1


# --- The command line ---------------------------------------------------------------------------------


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def base_args(tmp_path):
    return ["--dataset", str(DATASET_PATH), "--results-dir", str(tmp_path / "results")]


def full_script(reps=1):
    return [r for _ in range(reps) for case in DATASET.cases for r in oracle_script(case.id)]


def test_full_run_with_known_good_answers_scores_100_percent(isolated, capsys):
    code = evals.main(
        [*base_args(isolated), "--engine", "handwritten"],
        client=FakeClient(full_script()),
    )

    out = capsys.readouterr().out
    assert code == 0
    assert f"{len(CASES)}/{len(CASES)}" in out and "100%" in out
    assert "FAIL" not in out
    assert len(list((isolated / "results").glob("*.json"))) == 1


def test_both_engines_run_through_the_command(isolated, capsys):
    script = full_script()
    code = evals.main(
        [*base_args(isolated), "--engine", "both"],
        client=FakeClient(script),
        chat_model_factory=_shared_scripted_model(full_script()),
    )

    out = capsys.readouterr().out
    assert code == 0
    assert out.count(f"{len(CASES)}/{len(CASES)}") == 2
    assert "handwritten" in out and "langgraph" in out


def _shared_scripted_model(script):
    model = scripted_model(script)
    return lambda: model


def test_engine_order_alternates_so_neither_always_gets_the_warm_cache(isolated, capsys):
    cases = "missing-city,uncovered-city,list-shortlist,no-tools-needed"  # the dataset's order
    ids = cases.split(",")
    script = lambda: [r for case_id in ids for r in oracle_script(case_id)]  # noqa: E731

    evals.main(
        [*base_args(isolated), "--engine", "both", "--cases", cases],
        client=FakeClient(script()),
        chat_model_factory=_shared_scripted_model(script()),
    )

    lines = [line.split() for line in capsys.readouterr().out.splitlines() if line.startswith("  PASS")]
    first_engine_per_case = [lines[i][1] for i in range(0, len(lines), 2)]
    assert first_engine_per_case == ["handwritten", "langgraph", "handwritten", "langgraph"]
    # Each pair is the same case, and each engine runs every case once.
    assert all(lines[i][2] == lines[i + 1][2] for i in range(0, len(lines), 2))


def test_report_says_how_to_compare_costs_only_for_two_engines(isolated, capsys):
    usage = TokenUsage(input_tokens=10, output_tokens=10, requests=1)
    one = evals.summarize("a", [RunRecord("x", "a", usage=usage, checks=[Check("c", True)])])
    assert "No-cache/q." not in evals.format_report([one])
    assert "Compare engines with No-cache/q." in evals.format_report([one, one])


def test_reps_repeat_each_case(isolated, capsys):
    client = FakeClient(full_script(reps=2))
    assert evals.main([*base_args(isolated), "--engine", "handwritten", "--reps", "2"], client=client) == 0
    assert f"{len(CASES) * 2}/{len(CASES) * 2}" in capsys.readouterr().out


def test_effort_defaults_to_low_and_can_be_set(isolated):
    client = FakeClient(oracle_script("missing-city"))
    evals.main([*base_args(isolated), "--engine", "handwritten", "--cases", "missing-city"], client=client)
    assert client.requests[0]["output_config"] == {"effort": "low"}

    client = FakeClient(oracle_script("missing-city"))
    evals.main([*base_args(isolated), "--engine", "handwritten", "--cases", "missing-city", "--effort", "high"], client=client)
    assert client.requests[0]["output_config"] == {"effort": "high"}


def test_cases_flag_runs_only_those_cases(isolated, capsys):
    client = FakeClient(oracle_script("missing-city"))
    assert evals.main([*base_args(isolated), "--engine", "handwritten", "--cases", "missing-city"], client=client) == 0
    out = capsys.readouterr().out
    assert "missing-city" in out and "city-search" not in out


def test_unknown_case_id_is_reported(isolated, capsys):
    assert evals.main([*base_args(isolated), "--cases", "nope"], client=FakeClient([])) == 2
    assert "nope" in capsys.readouterr().err


def test_list_prints_cases_without_needing_a_key(isolated, capsys):
    assert evals.main([*base_args(isolated), "--list"]) == 0
    assert "budget-filter" in capsys.readouterr().out


def test_missing_key_exits_with_instructions(isolated, capsys):
    assert evals.main(base_args(isolated)) == 2
    assert "No Claude API key found" in capsys.readouterr().err


def test_langsmith_flag_needs_a_key(isolated, capsys):
    assert evals.main([*base_args(isolated), "--langsmith"], client=FakeClient([])) == 2
    assert "LANGSMITH_API_KEY" in capsys.readouterr().err


def test_bad_api_key_stops_the_run_instead_of_repeating_the_failure(isolated, capsys):
    client = FailingClient(api_error(anthropic.AuthenticationError, 401))
    assert evals.main([*base_args(isolated), "--engine", "handwritten"], client=client) == 1
    err = capsys.readouterr().err
    assert "stopping the eval" in err and "AuthenticationError" in err


def test_rate_limit_is_recorded_and_the_run_continues(isolated, capsys):
    client = FailingClient(api_error(anthropic.RateLimitError, 429))
    assert evals.main([*base_args(isolated), "--engine", "handwritten"], client=client) == 0
    out = capsys.readouterr().out
    assert "ERROR" in out and f"{len(CASES)} API errors" in out


def test_config_defaults_match_the_app():
    assert AgentConfig().model == evals.DEFAULT_MODEL


def test_build_source_uses_the_bundled_events(tmp_path):
    source = evals.build_source(DATASET, tmp_path)
    assert isinstance(source, SampleEventSource)
    assert len(source.search(city="Chicago", max_results=50)) == 7
