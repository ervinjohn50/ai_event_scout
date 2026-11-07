"""Event source backed by the Ticketmaster Discovery API (v2).

Docs: https://developer.ticketmaster.com/products-and-docs/apis/discovery-api/v2/
A free API key gives 5,000 calls/day at up to 5 requests/second.

Two jobs live here:
1. Build the HTTP request from the search filters.
2. Convert Ticketmaster's deeply nested JSON into the flat `Event` model.
   Real-world API data is messy (fields can be missing or null), so the parser never
   assumes a field exists or has the expected type. Anything it can't read is treated
   as unknown, so one odd record can't fail a whole search.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date, time
from typing import Any

from .base import EventSourceError
from .models import Event

BASE_URL = "https://app.ticketmaster.com/discovery/v2"

# The app's category words -> Ticketmaster "segment" names.
CATEGORY_TO_SEGMENT = {
    "music": "Music",
    "sports": "Sports",
    "arts": "Arts & Theatre",
    "film": "Film",
    "misc": "Miscellaneous",
}
SEGMENT_TO_CATEGORY = {segment.casefold(): cat for cat, segment in CATEGORY_TO_SEGMENT.items()}

# A "fetch" function takes a URL and a timeout and returns the response body.
# Passing it in (instead of calling urllib directly) lets tests swap in a fake one.
Fetch = Callable[[str, float], bytes]


def _urllib_fetch(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class TicketmasterEventSource:
    name = "ticketmaster"

    def __init__(
        self, api_key: str, fetch: Fetch = _urllib_fetch, timeout: float = 10.0, today: date | None = None
    ) -> None:
        if not api_key:
            raise ValueError("A Ticketmaster API key is required")
        self._api_key = api_key
        self._fetch = fetch
        self._timeout = timeout
        self._today = today

    def describe(self) -> str:
        return "live data from the Ticketmaster Discovery API (real upcoming events worldwide)"

    def search(
        self,
        *,
        keyword: str | None = None,
        city: str | None = None,
        country_code: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        category: str | None = None,
        max_results: int = 10,
    ) -> list[Event]:
        params: dict[str, Any] = {"size": max_results, "sort": "date,asc"}
        if keyword:
            params["keyword"] = keyword
        if city:
            params["city"] = city
        if country_code:
            params["countryCode"] = country_code.upper()
        if category:
            params["segmentName"] = CATEGORY_TO_SEGMENT[category]
        # Filter on the venue's local time. The API's startDateTime and endDateTime are in
        # UTC, so "Saturday" in Chicago would mean Friday evening through Saturday afternoon,
        # which returned mostly Friday's shows and missed Saturday's concerts. Without a
        # start date the API also returns events from past years, so it defaults to today.
        start = start_date or self._today or date.today()
        end = f"{end_date.isoformat()}T23:59:59" if end_date else "*"
        params["localStartDateTime"] = f"{start.isoformat()}T00:00:00,{end}"

        data = self._get("/events.json", params)
        raw_events = _dict(data.get("_embedded")).get("events") or []
        return [parse_event(raw) for raw in raw_events if _has_id(raw)]

    def get(self, event_id: str) -> Event | None:
        try:
            data = self._get(f"/events/{urllib.parse.quote(event_id, safe='')}.json", {})
        except _NotFound:
            return None
        return parse_event(data) if _has_id(data) else None

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode({**params, "apikey": self._api_key})
        url = f"{BASE_URL}{path}?{query}"
        try:
            body = self._fetch(url, self._timeout)
        except urllib.error.HTTPError as err:
            # Careful: never put `url` in these messages. It contains the API key, and
            # these messages are sent to the model as tool results.
            if err.code == 404:
                raise _NotFound() from None
            if err.code == 401:
                raise EventSourceError("Ticketmaster rejected the API key (HTTP 401).") from None
            if err.code == 429:
                raise EventSourceError(
                    "Ticketmaster rate limit reached (HTTP 429). Wait a moment and retry."
                ) from None
            raise EventSourceError(f"Ticketmaster returned HTTP {err.code}.") from None
        except (urllib.error.URLError, TimeoutError) as err:
            reason = getattr(err, "reason", err)
            raise EventSourceError(f"Could not reach Ticketmaster: {reason}") from None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            raise EventSourceError("Ticketmaster returned a response that isn't valid JSON.") from None


class _NotFound(Exception):
    pass


def parse_event(raw: dict[str, Any]) -> Event:
    """Flatten one Ticketmaster event object into an `Event`. The object must have an id."""
    dates = _dict(raw.get("dates"))
    start = _dict(dates.get("start"))
    embedded = _dict(raw.get("_embedded"))
    venue = _first(embedded.get("venues"))
    classification = _first(raw.get("classifications"))
    # A range of 0 to 0 is a placeholder on events with no price set, and doesn't mean free.
    prices = [
        p
        for p in map(_dict, raw.get("priceRanges") or [])
        if _amount(p.get("min")) or _amount(p.get("max"))
    ]
    segment = _dict(classification.get("segment")).get("name") or ""
    genre = _dict(classification.get("genre")).get("name")

    return Event(
        id=raw["id"],
        name=raw.get("name") or "Untitled event",
        source="ticketmaster",
        start_date=_parse_date(start.get("localDate")),
        start_time=_parse_time(start.get("localTime")),
        category=SEGMENT_TO_CATEGORY.get(segment.casefold()),
        # Ticketmaster uses "Undefined" as a placeholder; treat it as missing.
        genre=genre if genre and genre != "Undefined" else None,
        venue=venue.get("name"),
        venue_address=_dict(venue.get("address")).get("line1"),
        city=_dict(venue.get("city")).get("name"),
        state_code=_dict(venue.get("state")).get("stateCode"),
        country_code=_dict(venue.get("country")).get("countryCode"),
        price_min=min((p["min"] for p in prices if _amount(p.get("min")) is not None), default=None),
        price_max=max((p["max"] for p in prices if _amount(p.get("max")) is not None), default=None),
        currency=next((p["currency"] for p in prices if p.get("currency")), None),
        status=_dict(dates.get("status")).get("code"),
        url=raw.get("url"),
        lineup=tuple(
            name for attraction in embedded.get("attractions") or [] if (name := _dict(attraction).get("name"))
        ),
    )


def _has_id(raw: Any) -> bool:
    """Without an id, the agent can't look an event up or save it, so it's left out."""
    return isinstance(raw, dict) and isinstance(raw.get("id"), str) and bool(raw["id"])


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first(items: list[Any] | None) -> dict[str, Any]:
    return _dict(items[0]) if items else {}


def _amount(value: Any) -> float | None:
    # bool counts as a number in Python, but true isn't a price.
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


def _parse_time(value: Any) -> time | None:
    try:
        return time.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None
