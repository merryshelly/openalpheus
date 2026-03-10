"""SQLite schema creation and sqlite-vec loading."""
import sqlite3
from pathlib import Path


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
        
        # Enable extension loading
        conn.enable_load_extension(True)
        
        # Load sqlite-vec extension
        sqlite_vec.load(conn)
        
        # Get dimensions from config table, default to 768 if not found
        cursor = conn.execute("SELECT value FROM _config WHERE key = 'dimensions'")
        row = cursor.fetchone()
        dimensions = int(row[0]) if row else 768
        
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
