"""Event source backed by the Ticketmaster Discovery API (v2).

Docs: https://developer.ticketmaster.com/products-and-docs/apis/discovery-api/v2/
A free API key gives 5,000 calls/day at up to 5 requests/second.

Two jobs live here:
1. Build the HTTP request from our search filters.
2. Convert Ticketmaster's deeply nested JSON into our flat `Event` model.
   Real-world API data is messy (fields go missing all the time), so the parser
   uses `.get()` everywhere and never assumes a field exists.
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

# Our category words -> Ticketmaster "segment" names.
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

    def __init__(self, api_key: str, fetch: Fetch = _urllib_fetch, timeout: float = 10.0) -> None:
        if not api_key:
            raise ValueError("A Ticketmaster API key is required")
        self._api_key = api_key
        self._fetch = fetch
        self._timeout = timeout

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
        # Ticketmaster wants full UTC timestamps without fractional seconds.
        if start_date:
            params["startDateTime"] = f"{start_date.isoformat()}T00:00:00Z"
        if end_date:
            params["endDateTime"] = f"{end_date.isoformat()}T23:59:59Z"

        data = self._get("/events.json", params)
        raw_events = data.get("_embedded", {}).get("events", [])
        return [parse_event(raw) for raw in raw_events]

    def get(self, event_id: str) -> Event | None:
        try:
            data = self._get(f"/events/{urllib.parse.quote(event_id, safe='')}.json", {})
        except _NotFound:
            return None
        return parse_event(data)

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
    """Flatten one Ticketmaster event object into our `Event` model."""
    start = raw.get("dates", {}).get("start", {})
    venue = _first(raw.get("_embedded", {}).get("venues"))
    classification = _first(raw.get("classifications"))
    prices = raw.get("priceRanges") or []
    segment = (classification.get("segment") or {}).get("name", "")
    genre = (classification.get("genre") or {}).get("name")

    return Event(
        id=raw["id"],
        name=raw.get("name", "Untitled event"),
        source="ticketmaster",
        start_date=_parse_date(start.get("localDate")),
        start_time=_parse_time(start.get("localTime")),
        category=SEGMENT_TO_CATEGORY.get(segment.casefold()),
        # Ticketmaster uses "Undefined" as a placeholder; treat it as missing.
        genre=genre if genre and genre != "Undefined" else None,
        venue=venue.get("name"),
        venue_address=(venue.get("address") or {}).get("line1"),
        city=(venue.get("city") or {}).get("name"),
        state_code=(venue.get("state") or {}).get("stateCode"),
        country_code=(venue.get("country") or {}).get("countryCode"),
        price_min=min((p["min"] for p in prices if "min" in p), default=None),
        price_max=max((p["max"] for p in prices if "max" in p), default=None),
        currency=prices[0].get("currency") if prices else None,
        status=(raw.get("dates", {}).get("status") or {}).get("code"),
        url=raw.get("url"),
        lineup=tuple(a["name"] for a in raw.get("_embedded", {}).get("attractions", []) if "name" in a),
    )


def _first(items: list[dict[str, Any]] | None) -> dict[str, Any]:
    return items[0] if items else {}


def _parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _parse_time(value: str | None) -> time | None:
    try:
        return time.fromisoformat(value) if value else None
    except ValueError:
        return None
