"""Durable snapshot storage in SQLite (DESIGN.md §14).

One archive file holds many *sessions* (a session is one connect-to-
disconnect run on one game). Snapshots are stored one row per region so a
later query can pull a single region's history without decoding the rest.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    game        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS snapshots (
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    frame       INTEGER NOT NULL,
    region      TEXT NOT NULL,
    data        BLOB NOT NULL,
    PRIMARY KEY (session_id, seq, region)
);
CREATE INDEX IF NOT EXISTS snapshots_by_session ON snapshots(session_id, seq);
"""


class SnapshotArchive:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if self._path.exists() and self._path.is_dir():
            raise ConfigurationError(f"{self._path} is a folder, not a database file.")
        try:
            self._db = sqlite3.connect(str(self._path), isolation_level=None)
        except sqlite3.Error as exc:
            raise ConfigurationError(f"Can't open snapshot archive {self._path}: {exc}") from exc
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        # WAL + NORMAL never corrupts the file; at worst the last few snapshots
        # before a power cut are lost. FULL costs one fsync per snapshot, which
        # on Windows runners took ~11 ms each and made recording stutter.
        self._db.execute("PRAGMA synchronous = NORMAL")
        self._db.executescript(_SCHEMA)
        self._seq: dict[int, int] = {}

    # ----------------------------------------------------------- lifecycle

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> SnapshotArchive:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------ sessions

    def new_session(self, name: str, game: str) -> int:
        if not name.strip() or not game.strip():
            raise ConfigurationError("A session needs a name and a game.")
        cur = self._db.execute("INSERT INTO sessions(name, game) VALUES (?, ?)", (name, game))
        session_id = cur.lastrowid
        assert session_id is not None
        self._seq[session_id] = 0
        return session_id

    def sessions(self) -> list[tuple[int, str, str]]:
        rows = self._db.execute("SELECT id, name, game FROM sessions ORDER BY id").fetchall()
        return [(int(i), str(n), str(g)) for i, n, g in rows]

    def delete_session(self, session_id: int) -> None:
        self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._seq.pop(session_id, None)

    # ----------------------------------------------------------- snapshots

    def append(self, session_id: int, snapshot: MemorySnapshot) -> int:
        """Store ``snapshot``; returns its sequence number within the session."""
        if session_id not in self._seq:
            row = self._db.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM snapshots WHERE session_id = ?", (session_id,)
            ).fetchone()
            exists = self._db.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                raise ConfigurationError(f"No session {session_id} in the archive.")
            self._seq[session_id] = int(row[0]) + 1
        seq = self._seq[session_id]
        self._db.execute("BEGIN")
        try:
            self._db.executemany(
                "INSERT INTO snapshots(session_id, seq, frame, region, data) "
                "VALUES (?, ?, ?, ?, ?)",
                [(session_id, seq, snapshot.frame, n, d) for n, d in snapshot.regions.items()],
            )
            self._db.execute("COMMIT")
        except sqlite3.Error:
            self._db.execute("ROLLBACK")
            raise
        self._seq[session_id] = seq + 1
        return seq

    def count(self, session_id: int) -> int:
        row = self._db.execute(
            "SELECT COUNT(DISTINCT seq) FROM snapshots WHERE session_id = ?", (session_id,)
        ).fetchone()
        return int(row[0])

    def iter_session(self, session_id: int) -> Iterator[MemorySnapshot]:
        rows = self._db.execute(
            "SELECT seq, frame, region, data FROM snapshots WHERE session_id = ? "
            "ORDER BY seq, region",
            (session_id,),
        )
        current_seq: int | None = None
        frame = 0
        regions: dict[str, bytes] = {}
        for seq, fr, region, data in rows:
            if current_seq is not None and seq != current_seq:
                yield MemorySnapshot(frame, regions)
                regions = {}
            current_seq, frame = seq, int(fr)
            regions[str(region)] = bytes(data)
        if current_seq is not None:
            yield MemorySnapshot(frame, regions)

    def latest(self, session_id: int) -> MemorySnapshot | None:
        row = self._db.execute(
            "SELECT MAX(seq) FROM snapshots WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row[0] is None:
            return None
        rows = self._db.execute(
            "SELECT frame, region, data FROM snapshots WHERE session_id = ? AND seq = ?",
            (session_id, row[0]),
        ).fetchall()
        return MemorySnapshot(int(rows[0][0]), {str(r): bytes(d) for _, r, d in rows})

    def region_history(self, session_id: int, region: str) -> list[tuple[int, bytes]]:
        """``(frame, data)`` for one region across the session, oldest first."""
        rows = self._db.execute(
            "SELECT frame, data FROM snapshots WHERE session_id = ? AND region = ? ORDER BY seq",
            (session_id, region),
        ).fetchall()
        return [(int(f), bytes(d)) for f, d in rows]


__all__ = ["SnapshotArchive"]
