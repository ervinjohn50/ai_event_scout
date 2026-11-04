"""Tests for the tools layer: schemas, validation, error handling, and the shortlist.

These call `ToolRegistry.run` exactly the way the agent loop will, with a tool name
and a raw dict of input, just as the model would send it.
"""

import json
from datetime import date

import pytest

from event_scout.events import EventSourceError, SampleEventSource
from event_scout.shortlist import ShortlistError, ShortlistStore
from event_scout.tools import build_tools

TODAY = date(2026, 11, 3)


@pytest.fixture
def store(tmp_path) -> ShortlistStore:
    return ShortlistStore(tmp_path / "shortlist.json")


@pytest.fixture
def registry(store):
    return build_tools(SampleEventSource(today=TODAY), store)


def run_ok(registry, name, tool_input):
    """Run a tool, assert it succeeded, and return the parsed JSON result."""
    outcome = registry.run(name, tool_input)
    assert not outcome.is_error, outcome.content
    return json.loads(outcome.content)


# --- Definitions: what Claude sees ---------------------------------------------------------


def test_definitions_list_every_tool_in_a_stable_order(registry):
    names = [d["name"] for d in registry.definitions()]
    assert names == ["search_events", "get_event_details", "save_to_shortlist", "list_shortlist"]
    assert registry.definitions() == registry.definitions()  # identical on every call


def test_definitions_have_descriptions_and_object_schemas(registry):
    for definition in registry.definitions():
        assert set(definition) == {"name", "description", "input_schema"}
        assert len(definition["description"]) > 40
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert "title" not in json.dumps(schema)  # pydantic's noisy titles are stripped


def test_search_schema_exposes_constraints(registry):
    props = registry.definitions()[0]["input_schema"]["properties"]
    assert props["max_results"]["maximum"] == 25
    category = next(option for option in props["category"]["anyOf"] if "enum" in option)
    assert category["enum"] == ["music", "sports", "arts", "film", "misc"]
    assert "description" in props["city"]


# --- search_events -----------------------------------------------------------------------


def test_search_returns_compact_summaries(registry):
    result = run_ok(registry, "search_events", {"city": "Chicago", "category": "sports"})
    assert result["count"] == 1
    event = result["events"][0]
    assert event["id"] == "sample-003"
    assert "lineup" not in event and "ticket_url" not in event  # details stay out of search


def test_search_accepts_dates_as_strings(registry):
    result = run_ok(
        registry, "search_events", {"city": "Chicago", "start_date": "2026-11-10", "end_date": "2026-11-16"}
    )
    # Nov 10 (+7 days), Nov 12 (+9), Nov 15 (+12): all inside the window
    assert [e["id"] for e in result["events"]] == ["sample-003", "sample-004", "sample-005"]


def test_search_with_no_results_includes_a_hint(registry):
    result = run_ok(registry, "search_events", {"city": "Atlantis"})
    assert result["count"] == 0
    assert "fewer filters" in result["hint"]


def test_search_that_hits_max_results_says_there_may_be_more(registry):
    result = run_ok(registry, "search_events", {"max_results": 2})
    assert result["count"] == 2 and "note" in result


# --- Validation: the model sends bad input ------------------------------------------------


def test_out_of_range_value_is_an_error_naming_the_field(registry):
    outcome = registry.run("search_events", {"max_results": 500})
    assert outcome.is_error
    assert "max_results" in outcome.content and "500" in outcome.content


def test_invented_field_is_rejected(registry):
    outcome = registry.run("search_events", {"town": "Chicago"})
    assert outcome.is_error and "town" in outcome.content


def test_end_date_before_start_date_is_rejected(registry):
    outcome = registry.run("search_events", {"start_date": "2026-12-01", "end_date": "2026-11-01"})
    assert outcome.is_error and "end_date must be on or after start_date" in outcome.content


def test_missing_required_field_is_rejected(registry):
    outcome = registry.run("get_event_details", {})
    assert outcome.is_error and "event_id" in outcome.content


def test_unknown_tool_lists_the_real_ones(registry):
    outcome = registry.run("buy_tickets", {})
    assert outcome.is_error
    assert "buy_tickets" in outcome.content and "search_events" in outcome.content


# --- get_event_details --------------------------------------------------------------------


def test_get_event_details_includes_lineup_and_ticket_url(registry):
    details = run_ok(registry, "get_event_details", {"event_id": "sample-001"})
    assert details["lineup"] == ["Velvet Static", "Kestrel Moon"]
    assert details["ticket_url"].startswith("https://")
    assert details["price"] == "25-40 USD"


def test_get_event_details_unknown_id_is_an_error(registry):
    outcome = registry.run("get_event_details", {"event_id": "sample-999"})
    assert outcome.is_error and "sample-999" in outcome.content


# --- Shortlist ----------------------------------------------------------------------------


def test_save_and_list_shortlist(registry):
    saved = run_ok(registry, "save_to_shortlist", {"event_id": "sample-002", "note": "date night"})
    assert saved["status"] == "saved" and saved["shortlist_size"] == 1

    listed = run_ok(registry, "list_shortlist", {})
    assert listed["count"] == 1
    assert listed["entries"][0]["name"] == "Lake Effect Brass Band"
    assert listed["entries"][0]["note"] == "date night"


def test_saving_twice_updates_the_note_instead_of_duplicating(registry):
    run_ok(registry, "save_to_shortlist", {"event_id": "sample-002"})
    again = run_ok(registry, "save_to_shortlist", {"event_id": "sample-002", "note": "bring Sam"})
    assert again["status"] == "already_saved"
    assert again["shortlist_size"] == 1
    assert again["entry"]["note"] == "bring Sam"


def test_shortlist_survives_a_restart(registry, store):
    run_ok(registry, "save_to_shortlist", {"event_id": "sample-002"})
    # A brand new store reading the same file, like starting the app again tomorrow.
    assert [e["event_id"] for e in ShortlistStore(store.path).all()] == ["sample-002"]


def test_save_unknown_event_is_an_error(registry, store):
    outcome = registry.run("save_to_shortlist", {"event_id": "made-up"})
    assert outcome.is_error
    assert store.all() == []


def test_empty_shortlist(registry):
    assert run_ok(registry, "list_shortlist", {}) == {"count": 0, "entries": []}


def test_corrupted_shortlist_file_is_reported_not_crashed(registry, store):
    store.path.write_text("{not json")
    outcome = registry.run("list_shortlist", {})
    assert outcome.is_error and "not valid JSON" in outcome.content
    with pytest.raises(ShortlistError):
        store.all()


# --- Failures inside tools never crash the agent -------------------------------------------


class FailingSource(SampleEventSource):
    def search(self, **kwargs):
        raise EventSourceError("Ticketmaster rate limit reached (HTTP 429).")


class BuggySource(SampleEventSource):
    def get(self, event_id):
        raise KeyError("oops")  # simulates a bug in our own code


def test_data_source_error_becomes_an_error_result(store):
    registry = build_tools(FailingSource(today=TODAY), store)
    outcome = registry.run("search_events", {"city": "Chicago"})
    assert outcome.is_error and "rate limit" in outcome.content


def test_unexpected_exception_becomes_an_error_result(store):
    registry = build_tools(BuggySource(today=TODAY), store)
    outcome = registry.run("get_event_details", {"event_id": "sample-001"})
    assert outcome.is_error
    assert outcome.content.startswith("Internal error in get_event_details: KeyError")
