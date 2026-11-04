"""The normalized Event model.

Every data source (Ticketmaster, the offline sample file, anything added later)
converts its own format into this one shape. The tools and the agent only ever see
`Event`, so swapping data sources never touches the agent code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Any, Literal

# The small, fixed set of categories the agent can filter on. Keeping this list short
# (and using the same words in the tool schema) makes it easy for the model to pick
# the right one.
Category = Literal["music", "sports", "arts", "film", "misc"]
CATEGORIES: tuple[str, ...] = ("music", "sports", "arts", "film", "misc")


@dataclass(frozen=True)
class Event:
    id: str
    name: str
    source: str  # "ticketmaster" or "sample": useful when debugging where data came from
    start_date: date | None = None  # None when the source says "date to be announced"
    start_time: time | None = None
    category: str | None = None
    genre: str | None = None
    venue: str | None = None
    venue_address: str | None = None
    city: str | None = None
    state_code: str | None = None
    country_code: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    currency: str | None = None
    status: str | None = None  # e.g. "onsale", "offsale", "cancelled", "rescheduled"
    url: str | None = None
    lineup: tuple[str, ...] = ()

    def price_label(self) -> str | None:
        """Human-readable price, e.g. '45-95 USD' or '30 GBP'. None when unknown."""
        if self.price_min is None and self.price_max is None:
            return None
        low = self.price_min if self.price_min is not None else self.price_max
        high = self.price_max if self.price_max is not None else self.price_min
        currency = f" {self.currency}" if self.currency else ""
        if low == high:
            return f"{_fmt_money(low)}{currency}"
        return f"{_fmt_money(low)}-{_fmt_money(high)}{currency}"

    def summary(self) -> dict[str, Any]:
        """Compact view used in search results.

        Why compact? Every tool result is appended to the conversation and re-sent to
        the model on every later step. Short results keep the context small, which
        keeps the agent fast and cheap. Full details are one `get_event_details`
        call away.
        """
        return _drop_none(
            {
                "id": self.id,
                "name": self.name,
                "date": self.start_date.isoformat() if self.start_date else "TBA",
                "time": self.start_time.strftime("%H:%M") if self.start_time else None,
                "venue": self.venue,
                "city": self.city,
                "category": self.category,
                "genre": self.genre,
                "price": self.price_label(),
                "status": self.status,
            }
        )

    def details(self) -> dict[str, Any]:
        """Every known detail about the event, for `get_event_details`."""
        return _drop_none(
            {
                **self.summary(),
                "lineup": list(self.lineup) or None,
                "venue_address": self.venue_address,
                "state_code": self.state_code,
                "country_code": self.country_code,
                "ticket_url": self.url,
                "source": self.source,
            }
        )


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}"


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    # Leaving out empty fields saves tokens and avoids the model reading "null" as
    # meaningful information.
    return {k: v for k, v in d.items() if v is not None}
