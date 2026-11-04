"""Tests for the event data sources. No network, no API keys, no AI."""

import json
import urllib.error
import urllib.parse
from datetime import date, time, timedelta
from pathlib import Path

import pytest

from event_scout.events import EventSourceError, SampleEventSource, TicketmasterEventSource
from event_scout.events.ticketmaster import parse_event

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 11, 3)


# --- Offline sample source -----------------------------------------------------------


@pytest.fixture
def sample() -> SampleEventSource:
    # Pin "today" so the relative dates in the sample file are predictable.
    return SampleEventSource(today=TODAY)


def test_sample_dates_are_relative_to_today(sample):
    event = sample.get("sample-001")  # days_from_today: 3
    assert event.start_date == TODAY + timedelta(days=3)
    assert event.start_time == time(20, 0)


def test_sample_results_are_sorted_soonest_first(sample):
    dates = [e.start_date for e in sample.search(max_results=50)]
    assert dates == sorted(dates)


def test_sample_city_filter_is_case_insensitive(sample):
    results = sample.search(city="chicago", max_results=50)
    assert results
    assert {e.city for e in results} == {"Chicago"}


def test_sample_category_and_date_range(sample):
    results = sample.search(
        city="Chicago",
        category="music",
        start_date=TODAY,
        end_date=TODAY + timedelta(days=14),
    )
    assert [e.id for e in results] == ["sample-001", "sample-002", "sample-004", "sample-005"]


def test_sample_keyword_matches_every_word_across_fields(sample):
    # "indie rock" is in the genre field; "chicago" is in the city field.
    results = sample.search(keyword="Indie Rock chicago", max_results=50)
    assert {e.id for e in results} == {"sample-001", "sample-004", "sample-007"}


def test_sample_keyword_matches_lineup(sample):
    results = sample.search(keyword="kestrel moon", max_results=50)
    assert {e.id for e in results} == {"sample-001", "sample-025"}


def test_sample_max_results(sample):
    assert len(sample.search(max_results=3)) == 3


def test_sample_get_unknown_id_returns_none(sample):
    assert sample.get("nope") is None


def test_sample_describe_lists_covered_cities(sample):
    assert "Chicago" in sample.describe() and "London" in sample.describe()


# --- Ticketmaster: parsing -----------------------------------------------------------


def load_fixture() -> dict:
    return json.loads((FIXTURES / "ticketmaster_events.json").read_text())


def test_parse_full_ticketmaster_event():
    event = parse_event(load_fixture()["_embedded"]["events"][0])

    assert event.id == "vvG1aZ9KbEXAMPLE"
    assert event.start_date == date(2026, 11, 14)
    assert event.start_time == time(19, 30)
    assert event.category == "music"
    assert event.genre == "Rock"
    assert (event.venue, event.city, event.state_code, event.country_code) == (
        "Riverfront Theatre",
        "Chicago",
        "IL",
        "US",
    )
    # min/max are taken across all price ranges
    assert (event.price_min, event.price_max, event.currency) == (39.5, 104.25, "USD")
    assert event.lineup == ("Velvet Static", "Kestrel Moon")
    assert event.status == "onsale"


def test_parse_ticketmaster_event_with_missing_fields():
    event = parse_event(load_fixture()["_embedded"]["events"][1])

    assert event.name == "Mystery Show"
    assert event.start_date is None and event.start_time is None
    assert event.venue is None and event.city is None
    assert event.price_min is None and event.price_label() is None
    assert event.genre is None  # "Undefined" is treated as missing
    assert event.category == "misc"
    assert event.summary()["date"] == "TBA"


# --- Ticketmaster: HTTP requests (with a fake fetch, so no network) -------------------


class FakeFetch:
    """Records requested URLs and returns a canned response (or raises an error)."""

    def __init__(self, body: bytes = b"{}", error: Exception | None = None):
        self.body = body
        self.error = error
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float) -> bytes:
        self.urls.append(url)
        if self.error:
            raise self.error
        return self.body

    def last_params(self) -> dict[str, str]:
        return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.urls[-1]).query))


def test_ticketmaster_search_builds_query_and_parses_results():
    fetch = FakeFetch(body=(FIXTURES / "ticketmaster_events.json").read_bytes())
    source = TicketmasterEventSource("secret-key", fetch=fetch)

    events = source.search(
        keyword="velvet static",
        city="Chicago",
        country_code="us",
        start_date=date(2026, 11, 3),
        end_date=date(2026, 11, 30),
        category="music",
        max_results=5,
    )

    assert [e.id for e in events] == ["vvG1aZ9KbEXAMPLE", "Z7r9jZ1AEXAMPLE2"]
    assert fetch.urls[-1].startswith("https://app.ticketmaster.com/discovery/v2/events.json?")
    assert fetch.last_params() == {
        "apikey": "secret-key",
        "keyword": "velvet static",
        "city": "Chicago",
        "countryCode": "US",
        "segmentName": "Music",
        "startDateTime": "2026-11-03T00:00:00Z",
        "endDateTime": "2026-11-30T23:59:59Z",
        "size": "5",
        "sort": "date,asc",
    }


def test_ticketmaster_search_with_no_results():
    # When nothing matches, Ticketmaster leaves out "_embedded" entirely.
    source = TicketmasterEventSource("k", fetch=FakeFetch(body=b'{"page": {"totalElements": 0}}'))
    assert source.search(city="Nowhere") == []


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x/?apikey=secret-key", code, "err", hdrs=None, fp=None)


def test_ticketmaster_bad_key_raises_readable_error_without_leaking_key():
    source = TicketmasterEventSource("secret-key", fetch=FakeFetch(error=_http_error(401)))
    with pytest.raises(EventSourceError) as excinfo:
        source.search(city="Chicago")
    assert "401" in str(excinfo.value)
    assert "secret-key" not in str(excinfo.value)


def test_ticketmaster_rate_limit_error():
    source = TicketmasterEventSource("k", fetch=FakeFetch(error=_http_error(429)))
    with pytest.raises(EventSourceError, match="rate limit"):
        source.search()


def test_ticketmaster_network_error():
    source = TicketmasterEventSource("k", fetch=FakeFetch(error=urllib.error.URLError("no route")))
    with pytest.raises(EventSourceError, match="Could not reach Ticketmaster"):
        source.search()


def test_ticketmaster_get_unknown_event_returns_none():
    source = TicketmasterEventSource("k", fetch=FakeFetch(error=_http_error(404)))
    assert source.get("does-not-exist") is None
