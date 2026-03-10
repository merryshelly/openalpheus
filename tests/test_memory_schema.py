"""Tests for SQLite schema creation and sqlite-vec loading.

Interface contract:
    init_db(db_path: Path, dimensions: int) -> sqlite3.Connection
    load_vec_extension(conn: sqlite3.Connection) -> bool

init_db creates the database with chunks table, FTS5 virtual table,
triggers, and (if sqlite-vec loads) the vec0 virtual table.
Returns a connection with WAL mode enabled.

load_vec_extension attempts to load sqlite-vec and returns True/False.
"""

import sqlite3
import pytest
from pathlib import Path
from openalph.memory.schema import init_db, load_vec_extension


class TestInitDb:

    def test_creates_database_file(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path, dimensions=768)
        assert db_path.exists()
        conn.close()

    def test_returns_connection(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path, dimensions=768)
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_creates_chunks_table(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path, dimensions=768)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_chunks_table_has_correct_columns(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path, dimensions=768)
        cursor = conn.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {"id", "path", "start_line", "end_line", "text", "source",
                    "model", "file_mtime", "embedding"}
        assert expected.issubset(columns)
        conn.close()

    def test_creates_fts_table(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path, dimensions=768)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_fts_search_works(self, tmp_path):
        """FTS5 table is functional for text search."""
        db_path = tmp_path / "test.db"
        conn = init_db(db_path, dimensions=768)
        # Insert a chunk
        conn.execute(
            "INSERT INTO chunks (id, path, start_line, end_line, text, source, model, file_mtime) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("c1", "test.md", 1, 5, "validator monitoring ntfy alerts", "memory", "test-model", 1000.0)
        )
        conn.commit()
        # Search via FTS
        cursor = conn.execute(
            "SELECT id, text FROM chunks_fts WHERE chunks_fts MATCH ?",
            ('"validator"',)
        )
        results = cursor.fetchall()
        assert len(results) == 1
        assert results[0][0] == "c1"
        conn.close()

    def test_wal_mode_enabled(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path, dimensions=768)
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_creates_parent_directories(self, tmp_path):
        db_path = tmp_path / "subdir" / "deep" / "test.db"
        conn = init_db(db_path, dimensions=768)
        assert db_path.exists()
        conn.close()

    def test_idempotent(self, tmp_path):
        """Calling init_db twice on same path doesn't error or lose data."""
        db_path = tmp_path / "test.db"
        conn1 = init_db(db_path, dimensions=768)
        conn1.execute(
            "INSERT INTO chunks (id, path, start_line, end_line, text, source, model, file_mtime) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("c1", "test.md", 1, 1, "hello", "memory", "m", 1.0)
        )
        conn1.commit()
        conn1.close()

        conn2 = init_db(db_path, dimensions=768)
        cursor = conn2.execute("SELECT COUNT(*) FROM chunks")
        assert cursor.fetchone()[0] == 1
        conn2.close()


class TestLoadVecExtension:

    def test_returns_bool(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path, dimensions=768)
        result = load_vec_extension(conn)
        assert isinstance(result, bool)
        conn.close()

    def test_vec_table_created_on_success(self, tmp_path):
        """If sqlite-vec loads, the chunks_vec virtual table should exist."""
        db_path = tmp_path / "test.db"
        conn = init_db(db_path, dimensions=768)
        loaded = load_vec_extension(conn)
        if loaded:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
            )
            assert cursor.fetchone() is not None
        conn.close()

    def test_graceful_failure_if_not_installed(self, tmp_path):
        """If sqlite-vec is not available, returns False without raising."""
        db_path = tmp_path / "test.db"
        conn = init_db(db_path, dimensions=768)
        # Even if sqlite-vec IS installed, this test verifies the function
        # doesn't raise. The return value depends on the environment.
        result = load_vec_extension(conn)
        assert isinstance(result, bool)
        conn.close()
