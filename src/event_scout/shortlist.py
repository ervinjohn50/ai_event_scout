"""The user's personal shortlist of saved events, stored as a JSON file.

The model itself remembers nothing between conversations. Anything an agent should
"remember" has to live somewhere outside the model, and tools are how the agent
reads and writes it. This file is the simplest possible version of agent memory.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .events import Event

DEFAULT_SHORTLIST_PATH = Path("data") / "shortlist.json"


class ShortlistError(Exception):
    """The shortlist file exists but can't be read."""


class ShortlistStore:
    def __init__(self, path: Path = DEFAULT_SHORTLIST_PATH) -> None:
        self.path = Path(path)

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            entries = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise ShortlistError(f"The shortlist file {self.path} is not valid JSON.") from None
        if not isinstance(entries, list):
            raise ShortlistError(f"The shortlist file {self.path} has an unexpected format.")
        return entries

    def add(self, event: Event, note: str | None = None) -> tuple[dict[str, Any], bool]:
        """Save an event. Returns (entry, created): created is False if it was already saved."""
        entries = self.all()
        existing = next((e for e in entries if e["event_id"] == event.id), None)
        if existing is not None:
            # Saving again updates the note instead of creating a duplicate.
            if note:
                existing["note"] = note
            self._write(entries)
            return existing, False

        # Store a snapshot of the event, not just its ID: the shortlist should still
        # make sense later, even if the data source changes or the event disappears.
        entry = {
            "event_id": event.id,
            "name": event.name,
            "date": event.start_date.isoformat() if event.start_date else "TBA",
            "venue": event.venue,
            "city": event.city,
            "price": event.price_label(),
            "ticket_url": event.url,
            "source": event.source,
            "note": note,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        entries.append(entry)
        self._write(entries)
        return entry, True

    def _write(self, entries: list[dict[str, Any]]) -> None:
        # Write to a temporary file first, then swap it into place. If the program
        # crashes halfway through writing, the old shortlist is still intact.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
