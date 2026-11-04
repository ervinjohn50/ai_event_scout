"""The interface every event data source implements.

`EventSource` is a Protocol: any class with these methods counts as an event source.
The agent depends on this interface, not on Ticketmaster specifically. That's what
lets the tests (and the offline mode) swap in a different source without changing
any agent or tool code.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from .models import Event


class EventSourceError(Exception):
    """A data source failed (bad API key, rate limit, network down, ...).

    The message is shown to the model as a tool error, so it should explain the
    problem in plain words and must never contain secrets such as API keys.
    """


class EventSource(Protocol):
    name: str

    def describe(self) -> str:
        """One sentence for the system prompt, e.g. 'live Ticketmaster data'."""
        ...

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
        """Return upcoming events matching every filter that is set, soonest first."""
        ...

    def get(self, event_id: str) -> Event | None:
        """Return one event by ID, or None if it doesn't exist."""
        ...
