"""Offline event source backed by a bundled JSON file of fictional events.

Dates in the file are stored as `days_from_today` offsets rather than fixed dates,
so the sample data is always "upcoming" no matter when the project runs.
"""

from __future__ import annotations

import json
from datetime import date, time, timedelta
from pathlib import Path

from .models import Event

DEFAULT_SAMPLE_PATH = Path(__file__).with_name("sample_events.json")


class SampleEventSource:
    name = "sample"

    def __init__(self, path: Path = DEFAULT_SAMPLE_PATH, today: date | None = None) -> None:
        self.today = today or date.today()
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._events = [self._to_event(item) for item in raw]
        self._events.sort(key=lambda e: (e.start_date or date.max, e.start_time or time.min))

    def describe(self) -> str:
        cities = sorted({e.city for e in self._events if e.city})
        return (
            "the bundled OFFLINE SAMPLE dataset of fictional events. "
            f"It only covers these cities: {', '.join(cities)}"
        )

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
        start = start_date or self.today  # never return past events
        results = []
        for event in self._events:
            if event.start_date is None or event.start_date < start:
                continue
            if end_date and event.start_date > end_date:
                continue
            if city and (event.city or "").casefold() != city.casefold():
                continue
            if country_code and (event.country_code or "").casefold() != country_code.casefold():
                continue
            if category and event.category != category:
                continue
            if keyword and not _matches_keyword(event, keyword):
                continue
            results.append(event)
        return results[:max_results]

    def get(self, event_id: str) -> Event | None:
        return next((e for e in self._events if e.id == event_id), None)

    def _to_event(self, item: dict) -> Event:
        hour, minute = (int(part) for part in item["time"].split(":"))
        return Event(
            id=item["id"],
            name=item["name"],
            source=self.name,
            start_date=self.today + timedelta(days=item["days_from_today"]),
            start_time=time(hour, minute),
            category=item["category"],
            genre=item.get("genre"),
            venue=item["venue"],
            venue_address=item.get("venue_address"),
            city=item["city"],
            state_code=item.get("state_code"),
            country_code=item["country_code"],
            price_min=item.get("price_min"),
            price_max=item.get("price_max"),
            currency=item.get("currency"),
            status=item.get("status", "onsale"),
            url=item.get("url"),
            lineup=tuple(item.get("lineup", ())),
        )


def _matches_keyword(event: Event, keyword: str) -> bool:
    """Case-insensitive match against name, genre, venue, city, and lineup.

    Every word of the keyword must appear somewhere, so "indie rock" matches an
    event whose genre is "Indie Rock", and "jazz chicago" matches a Chicago jazz show.
    """
    haystack = " ".join(
        [event.name, event.genre or "", event.venue or "", event.city or "", *event.lineup]
    ).casefold()
    return all(word in haystack for word in keyword.casefold().split())
