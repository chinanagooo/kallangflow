"""SQLite persistence for attendee profiles.

Both demo.py (CLI) and app.py (Flask) import this module, so a profile
saved from one is immediately visible to the other — they share the
same attendees.db file, created automatically at the project root
(next to app.py) the first time init_db() runs.

"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import Attendee

# kallangflow/kallangflow/db.py -> parent.parent is the project root,
# same place app.py and demo.py are run from.
DB_PATH = Path(__file__).resolve().parent.parent / "attendees.db"


def init_db() -> None:
    """Create the attendees table if it doesn't exist yet. Safe to call
    every time the app/demo starts — CREATE TABLE IF NOT EXISTS is a no-op
    if the table's already there."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attendees (
                id TEXT PRIMARY KEY,
                home_location TEXT NOT NULL,
                transport_preference TEXT NOT NULL,
                accessibility_needs TEXT,
                travelling_with TEXT,
                event_start TEXT NOT NULL DEFAULT '19:30'
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_attendee(attendee: Attendee) -> None:
    """Insert a new profile, or update it in place if the id already
    exists (so re-running the demo/form with the same id overwrites
    rather than erroring)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO attendees
                (id, home_location, transport_preference, accessibility_needs, travelling_with, event_start)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                home_location=excluded.home_location,
                transport_preference=excluded.transport_preference,
                accessibility_needs=excluded.accessibility_needs,
                travelling_with=excluded.travelling_with,
                event_start=excluded.event_start
            """,
            (
                attendee.id,
                attendee.home_location,
                attendee.transport_preference,
                attendee.accessibility_needs,
                attendee.travelling_with,
                attendee.event_start,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_attendee(attendee_id: str) -> Attendee | None:
    """Look up a saved profile by id. Returns None if not found — the
    caller decides what default to fall back to."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            """
            SELECT id, home_location, transport_preference, accessibility_needs,
                   travelling_with, event_start
            FROM attendees WHERE id = ?
            """,
            (attendee_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return Attendee(
            id=row[0],
            home_location=row[1],
            transport_preference=row[2],
            accessibility_needs=row[3],
            travelling_with=row[4],
            event_start=row[5],
        )
    finally:
        conn.close()
