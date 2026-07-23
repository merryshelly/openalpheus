"""SQLite schema creation and sqlite-vec loading."""
import logging
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger("openalph.memory.schema")


def init_db(db_path: Path, dimensions: int) -> sqlite3.Connection:
    """Create/open the memory search database.
    
    Creates:
    - chunks table (id, path, start_line, end_line, text, source, model, file_mtime, embedding)
    - chunks_fts FTS5 virtual table (content-sync with chunks)
    - Triggers to keep FTS in sync with chunks table (INSERT and DELETE)
    - WAL journal mode
    - Parent directories if needed
    
    Returns an open connection. Idempotent (safe to call on existing DB).
    """
    # Create parent directories if needed
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open/create the database
    conn = sqlite3.connect(db_path)
    
    # Enable WAL mode
    conn.execute("PRAGMA journal_mode=WAL")
    
    # Create chunks table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            text TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'memory',
            model TEXT NOT NULL,
            file_mtime REAL NOT NULL,
            embedding BLOB
        )
    """)
    
    # Create FTS5 virtual table (content-sync with chunks)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            id UNINDEXED,
            path UNINDEXED,
            source UNINDEXED,
            start_line UNINDEXED,
            end_line UNINDEXED,
            model UNINDEXED,
            text,
            content='chunks',
            content_rowid='rowid'
        )
    """)
    
    # Create trigger for INSERT - sync to FTS
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks
        BEGIN
            INSERT INTO chunks_fts (rowid, id, path, source, start_line, end_line, model, text)
            VALUES (new.rowid, new.id, new.path, new.source, new.start_line, new.end_line, new.model, new.text);
        END
    """)
    
    # Create trigger for DELETE - remove from FTS
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks
        BEGIN
            INSERT INTO chunks_fts (chunks_fts, rowid, id, path, source, start_line, end_line, model, text)
            VALUES ('delete', old.rowid, old.id, old.path, old.source, old.start_line, old.end_line, old.model, old.text);
        END
    """)
    
    # Store dimensions for later use by load_vec_extension
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.execute(
        "INSERT OR REPLACE INTO _config (key, value) VALUES (?, ?)",
        ("dimensions", str(dimensions))
    )
    
    conn.commit()
    return conn


def load_vec_extension(conn: sqlite3.Connection) -> bool:
    """Try to load sqlite-vec extension and create the vec0 virtual table.
    
    Returns True if loaded successfully, False otherwise (graceful).
    On success, creates chunks_vec virtual table with the configured dimensions.
    """
    try:
        import sqlite_vec

        # Enable extension loading only for the duration of the load.
        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            # SEC-13: leaving extension loading enabled for the connection's
            # whole lifetime lets any later statement load an arbitrary shared
            # library. Disable it again the moment sqlite-vec is loaded; the
            # capability is needed for exactly that one call.
            conn.enable_load_extension(False)

        # Get dimensions from config table, default to 768 if not found
        cursor = conn.execute("SELECT value FROM _config WHERE key = 'dimensions'")
        row = cursor.fetchone()
        dimensions = int(row[0]) if row else 768

        # BUG-10: if a chunks_vec table already exists at a DIFFERENT width
        # (the operator changed the embedding model/dimension), CREATE ... IF
        # NOT EXISTS silently keeps the OLD width. Every new-width insert then
        # fails and -- with the indexer's per-row guard -- is dropped, leaving
        # a permanently keyword-only index while the operator believes semantic
        # search is live. Detect the mismatch against the ACTUAL table schema
        # and rebuild, so a dimension change self-heals on the next index pass.
        existing = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
        ).fetchone()
        if existing and existing[0]:
            m = re.search(r"float\[(\d+)\]", existing[0])
            existing_dim = int(m.group(1)) if m else None
            if existing_dim is not None and existing_dim != dimensions:
                logger.warning(
                    "Embedding dimension changed (%s -> %s); rebuilding chunks_vec.",
                    existing_dim, dimensions,
                )
                conn.execute("DROP TABLE IF EXISTS chunks_vec")

        # Create vec0 virtual table with the configured dimensions
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{dimensions}]
            )
        """)

        conn.commit()
    except Exception:
        return False

    return True
