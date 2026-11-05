"""The app's SQLite database: one file on disk, created the first time it's needed.

Each operation opens its own connection and closes it when done. That costs very little
with SQLite, and it means threads and separate programs never share a connection.
SQLite coordinates them through a lock on the file.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_DB_PATH = Path("data") / "event_scout.db"

# How long an operation waits for another writer to finish before giving up.
BUSY_TIMEOUT_SECONDS = 30

# Increase this whenever SCHEMA changes. SQLite stores the number inside the database
# file, so the app can tell whether a file matches this code.
SCHEMA_VERSION = 2

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS shortlist (
        event_id   TEXT PRIMARY KEY,
        name       TEXT NOT NULL,
        date       TEXT NOT NULL,
        venue      TEXT,
        city       TEXT,
        price      TEXT,
        ticket_url TEXT,
        source     TEXT NOT NULL,
        note       TEXT,
        saved_at   TEXT NOT NULL
    )
    """,
    # One row per question asked through the web API. It holds measurements only, not the
    # question or the answer, so usage reports don't keep what anyone asked.
    """
    CREATE TABLE IF NOT EXISTS question_runs (
        id                 INTEGER PRIMARY KEY,
        session_id         TEXT NOT NULL,
        engine             TEXT NOT NULL,
        model              TEXT NOT NULL,
        stop_reason        TEXT NOT NULL,
        steps              INTEGER NOT NULL,
        tool_calls         INTEGER NOT NULL,
        input_tokens       INTEGER NOT NULL,
        cache_read_tokens  INTEGER NOT NULL,
        cache_write_tokens INTEGER NOT NULL,
        output_tokens      INTEGER NOT NULL,
        cost_usd           REAL,
        seconds            REAL NOT NULL,
        created_at         TEXT NOT NULL
    )
    """,
]


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Open the database, creating the file and tables if needed, and close it afterward."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None turns off the sqlite3 module's automatic transactions, so a
    # transaction starts only where `transaction()` below starts one.
    connection = sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)
    try:
        connection.row_factory = sqlite3.Row  # lets each row convert to a dict
        _ensure_schema(connection)
        yield connection
    finally:
        connection.close()


@contextmanager
def transaction(path: Path) -> Iterator[sqlite3.Connection]:
    """Run several statements as one unit: all of them are saved, or none are.

    BEGIN IMMEDIATE takes the write lock at the start, so other writers wait their turn
    instead of working from data that's about to change.
    """
    with connect(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        with connection:  # commits at the end, or rolls back if an exception is raised
            yield connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    if _schema_version(connection) == SCHEMA_VERSION:
        return

    connection.execute("BEGIN IMMEDIATE")
    with connection:
        # Checked again with the lock held, in case another program just set it up.
        version = _schema_version(connection)
        if version > SCHEMA_VERSION:
            raise sqlite3.DatabaseError(
                f"the file uses schema version {version}, but this app only supports up to {SCHEMA_VERSION}"
            )
        for statement in SCHEMA:
            connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _schema_version(connection: sqlite3.Connection) -> int:
    return connection.execute("PRAGMA user_version").fetchone()[0]
