"""The user's personal shortlist of saved events, stored in the SQLite database.

The model itself remembers nothing between conversations. Anything an agent should
"remember" has to live somewhere outside the model, and tools are how the agent
reads and writes it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import DEFAULT_DB_PATH, connect, transaction
from .events import Event

_COLUMNS = "event_id, name, date, venue, city, price, ticket_url, source, note, saved_at"


class ShortlistError(Exception):
    """The database exists but can't be read or written."""


class ShortlistStore:
    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)

    def all(self) -> list[dict[str, Any]]:
        """Every saved event, oldest first."""
        with self._reporting_errors(), connect(self.path) as connection:
            rows = connection.execute(f"SELECT {_COLUMNS} FROM shortlist ORDER BY rowid").fetchall()
        return [dict(row) for row in rows]

    def add(self, event: Event, note: str | None = None) -> tuple[dict[str, Any], bool]:
        """Save an event. Returns (entry, created): created is False if it was already saved."""
        # Store a snapshot of the event, not just its ID: the shortlist should still
        # make sense later, even if the data source changes or the event disappears.
        values = {
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
        with self._reporting_errors(), transaction(self.path) as connection:
            # If the event_id is already saved, the database skips the insert. It checks this
            # itself, so two saves arriving together can't both create a row or overwrite
            # each other. Parameters (:name) keep event text from being run as SQL.
            inserted = connection.execute(
                """
                INSERT INTO shortlist (event_id, name, date, venue, city, price, ticket_url, source, note, saved_at)
                VALUES (:event_id, :name, :date, :venue, :city, :price, :ticket_url, :source, :note, :saved_at)
                ON CONFLICT (event_id) DO NOTHING
                """,
                values,
            )
            created = inserted.rowcount == 1
            # Saving again updates the note instead of creating a duplicate.
            if not created and note:
                connection.execute("UPDATE shortlist SET note = ? WHERE event_id = ?", (note, event.id))
            entry = connection.execute(
                f"SELECT {_COLUMNS} FROM shortlist WHERE event_id = ?", (event.id,)
            ).fetchone()
        return dict(entry), created

    def remove(self, event_id: str) -> bool:
        """Delete a saved event. Returns False if it wasn't saved."""
        with self._reporting_errors(), transaction(self.path) as connection:
            deleted = connection.execute("DELETE FROM shortlist WHERE event_id = ?", (event_id,))
            return deleted.rowcount == 1

    @contextmanager
    def _reporting_errors(self) -> Iterator[None]:
        """Turn storage failures into ShortlistError, which the tools report to the model."""
        try:
            yield
        except (sqlite3.Error, OSError) as err:
            raise ShortlistError(f"The database {self.path} can't be used: {err}.") from None
