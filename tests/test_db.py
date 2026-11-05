"""Tests for the SQLite database and the shortlist stored in it.

The concurrency tests use no lock in the test or in the store. The database alone has to
keep every save, including when separate programs write at the same time.
"""

import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from event_scout.db import SCHEMA_VERSION, connect, transaction
from event_scout.events import Event
from event_scout.shortlist import ShortlistError, ShortlistStore


def make_event(event_id: str = "event-1", name: str = "Test Show") -> Event:
    return Event(id=event_id, name=name, source="sample", venue="The Hall", city="Chicago")


@pytest.fixture
def path(tmp_path) -> Path:
    return tmp_path / "data" / "scout.db"


@pytest.fixture
def store(path) -> ShortlistStore:
    return ShortlistStore(path)


# --- The database ------------------------------------------------------------------------


def test_connecting_creates_the_folder_tables_and_version(path):
    with connect(path) as connection:
        tables = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert "shortlist" in tables
    assert version == SCHEMA_VERSION


def test_a_transaction_that_fails_saves_nothing(path, store):
    with pytest.raises(RuntimeError):
        with transaction(path) as connection:
            connection.execute(
                "INSERT INTO shortlist (event_id, name, date, source, saved_at) VALUES ('a', 'A', 'TBA', 'sample', 'now')"
            )
            raise RuntimeError("something went wrong partway through")
    assert store.all() == []


def test_a_database_from_a_newer_version_is_refused(path, store):
    with connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(ShortlistError, match="schema version"):
        store.all()


def test_a_folder_at_the_database_path_is_reported(tmp_path):
    with pytest.raises(ShortlistError):
        ShortlistStore(tmp_path).all()


# --- Saving and removing -----------------------------------------------------------------


def test_add_returns_the_entry_and_keeps_its_fields(store):
    entry, created = store.add(make_event(), "date night")
    assert created is True
    assert entry["event_id"] == "event-1"
    assert entry["date"] == "TBA"
    assert entry["note"] == "date night"
    assert entry["price"] is None
    assert store.all() == [entry]


def test_saving_again_without_a_note_keeps_the_old_note(store):
    store.add(make_event(), "date night")
    entry, created = store.add(make_event())
    assert created is False
    assert entry["note"] == "date night"


def test_saving_again_keeps_the_original_saved_at_and_details(store):
    first, _ = store.add(make_event(name="Original Name"))
    second, _ = store.add(make_event(name="Changed Name"), "a note")
    assert second["saved_at"] == first["saved_at"]
    assert second["name"] == "Original Name"


def test_entries_come_back_in_the_order_they_were_saved(store):
    for event_id in ("c", "a", "b"):
        store.add(make_event(event_id))
    assert [entry["event_id"] for entry in store.all()] == ["c", "a", "b"]


def test_remove(store):
    store.add(make_event("keep"))
    store.add(make_event("drop"))
    assert store.remove("drop") is True
    assert store.remove("drop") is False
    assert [entry["event_id"] for entry in store.all()] == ["keep"]


def test_text_that_looks_like_sql_is_stored_as_plain_text(store):
    name = "Robert'); DROP TABLE shortlist;--"
    entry, _ = store.add(make_event(name=name), note=name)
    assert entry["name"] == name and entry["note"] == name
    assert len(store.all()) == 1


# --- Saves happening at the same time ----------------------------------------------------


def test_simultaneous_saves_from_threads_all_survive(path):
    def save(number: int) -> None:
        ShortlistStore(path).add(make_event(f"event-{number}"))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(save, range(64)))

    assert len(ShortlistStore(path).all()) == 64


def test_saving_the_same_event_simultaneously_creates_one_row(path):
    def save(_: int) -> bool:
        return ShortlistStore(path).add(make_event("same"))[1]

    with ThreadPoolExecutor(max_workers=16) as pool:
        created = list(pool.map(save, range(32)))

    assert created.count(True) == 1
    assert len(ShortlistStore(path).all()) == 1


SAVE_SCRIPT = """
import sys
from event_scout.events import Event
from event_scout.shortlist import ShortlistStore

path, prefix = sys.argv[1], sys.argv[2]
for number in range(25):
    ShortlistStore(path).add(Event(id=f"{prefix}-{number}", name="Show", source="sample"))
"""


def test_simultaneous_saves_from_separate_programs_all_survive(path):
    """Like two terminals saving at once, which a lock inside one program can't protect."""
    programs = [
        subprocess.Popen([sys.executable, "-c", SAVE_SCRIPT, str(path), f"program{n}"], stderr=subprocess.PIPE)
        for n in range(4)
    ]
    for program in programs:
        _, errors = program.communicate(timeout=120)
        assert program.returncode == 0, errors.decode()

    assert len(ShortlistStore(path).all()) == 100


def test_reading_during_a_write_waits_instead_of_failing(path, store):
    store.add(make_event("first"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        saves = [pool.submit(store.add, make_event(f"event-{n}")) for n in range(16)]
        reads = [pool.submit(store.all) for _ in range(16)]
    for future in saves + reads:
        future.result()  # raises if any operation failed
    assert len(store.all()) == 17


def test_sqlite_errors_are_the_only_ones_converted(store, monkeypatch):
    """A bug in the app's own code should stay visible instead of looking like a storage problem."""
    def broken_connect(path):
        raise ValueError("a bug")

    monkeypatch.setattr("event_scout.shortlist.connect", broken_connect)
    with pytest.raises(ValueError):
        store.all()


def test_sqlite_error_type_is_converted(store, monkeypatch):
    def broken_connect(path):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr("event_scout.shortlist.connect", broken_connect)
    with pytest.raises(ShortlistError, match="disk I/O error"):
        store.all()
