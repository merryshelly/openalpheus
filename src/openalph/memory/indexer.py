"""File scanner, change detector, and indexing pipeline for memory search.

Ties together the chunker, embeddings, and schema modules.
"""

import asyncio
import hashlib
import logging
import os
import struct
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from openalph.memory.chunker import chunk_file

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    path: str      # absolute or relative file path
    mtime: float   # file modification time (os.path.getmtime)


@dataclass
class IndexStats:
    files_scanned: int
    files_indexed: int
    chunks_created: int
    files_removed: int


class MemoryIndexer:
    def __init__(self, db: sqlite3.Connection, embedder, model_name: str):
        """Initialize with a DB connection, embedding provider, and model name.
        
        db: SQLite connection from schema.init_db()
        embedder: EmbeddingProvider instance (has async embed_batch method)
        model_name: string stored in chunks.model column
        """
        self.db = db
        self.embedder = embedder
        self.model_name = model_name

    def _expected_dim(self) -> int | None:
        """Embedding width the vec table was built for (from _config), else None."""
        try:
            row = self.db.execute(
                "SELECT value FROM _config WHERE key = 'dimensions'"
            ).fetchone()
            return int(row[0]) if row else None
        except Exception:
            return None

    def scan_files(self, paths: list[Path]) -> list[FileInfo]:
        """Scan paths for .md files. Paths can be directories (recursive) or individual files.
        Skips missing paths silently. Returns FileInfo with absolute path and mtime."""
        files: list[FileInfo] = []
        
        for path in paths:
            if not path.exists():
                continue
            
            if path.is_dir():
                # Recursively find all .md files
                for md_file in path.rglob("*.md"):
                    if md_file.is_file():
                        files.append(FileInfo(
                            path=str(md_file),
                            mtime=os.path.getmtime(md_file)
                        ))
            elif path.is_file():
                # Include individual files (any extension)
                files.append(FileInfo(
                    path=str(path),
                    mtime=os.path.getmtime(path)
                ))
        
        return files

    def get_stale_files(self, files: list[FileInfo]) -> list[FileInfo]:
        """Compare file mtimes against DB. Returns files that need re-indexing.
        A file is stale if: not in DB at all, or DB mtime differs from current mtime."""
        # Query DB for existing files with their mtimes
        cursor = self.db.execute(
            "SELECT DISTINCT path, file_mtime FROM chunks WHERE model = ?",
            (self.model_name,)
        )
        db_files = {row[0]: row[1] for row in cursor.fetchall()}
        
        stale = []
        for file_info in files:
            if file_info.path not in db_files:
                # File not in DB - needs indexing
                stale.append(file_info)
            elif db_files[file_info.path] != file_info.mtime:
                # File mtime differs - needs re-indexing
                stale.append(file_info)
        
        return stale

    async def index_file(self, file_info: FileInfo) -> int:
        """Index a single file: read → chunk → embed → store in DB.
        First deletes any existing chunks for this path, then inserts new ones.
        Returns number of chunks created."""
        # BUG-9: read + chunk off the event loop, and never let one unreadable
        # file abort the whole index pass. `read_text` had no error handling, so
        # a single deleted / non-UTF-8 / permission-denied file raised straight
        # out of `index_all`. Treat such a file as "no chunks" and move on.
        def _read_and_chunk() -> list:
            text = Path(file_info.path).read_text(encoding="utf-8")
            return chunk_file(text, file_info.path)

        try:
            chunks = await asyncio.to_thread(_read_and_chunk)
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Skipping unreadable file %s: %s", file_info.path, e)
            return 0

        if not chunks:
            return 0

        # Embed all chunk texts
        texts = [c.text for c in chunks]
        embeddings = await self.embedder.embed_batch(texts)

        # Delete old chunks for this file (and their vec rows if the table exists)
        try:
            old_ids = [row[0] for row in self.db.execute(
                "SELECT id FROM chunks WHERE path = ? AND model = ?",
                (file_info.path, self.model_name)
            ).fetchall()]
            if old_ids:
                placeholders = ",".join("?" for _ in old_ids)
                self.db.execute(
                    f"DELETE FROM chunks_vec WHERE id IN ({placeholders})",
                    old_ids
                )
        except Exception:
            pass  # chunks_vec may not exist if sqlite-vec not loaded
        self.db.execute(
            "DELETE FROM chunks WHERE path = ? AND model = ?",
            (file_info.path, self.model_name)
        )

        # Validate embedding dimensions once. A wrong-width vector is rejected by
        # sqlite-vec or (worse) stored as an un-queryable row whose distance comes
        # back NULL and breaks vector search. Treat a malformed embedding as "no
        # embedding": the chunk is still indexed for keyword search, it just gets
        # no vector. Logged loudly so corruption is never silent again.
        expected_dim = self._expected_dim()
        emb_bytes_list: list[bytes | None] = []
        for idx in range(len(chunks)):
            emb = embeddings[idx] if idx < len(embeddings) else None
            if emb is None:
                emb_bytes_list.append(None)
            elif expected_dim is not None and len(emb) != expected_dim:
                logger.warning(
                    "Skipping malformed embedding for %s (chunk %d): got dim %d, expected %d",
                    file_info.path, idx, len(emb), expected_dim,
                )
                emb_bytes_list.append(None)
            else:
                emb_bytes_list.append(struct.pack(f"{len(emb)}f", *emb))

        # Insert new chunks
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(
                f"{file_info.path}:{chunk.start_line}:{i}".encode()
            ).hexdigest()
            emb_bytes = emb_bytes_list[i]

            self.db.execute(
                """INSERT INTO chunks
                   (id, path, start_line, end_line, text, source, model, file_mtime, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk_id,
                    chunk.path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.text,
                    "memory",  # source
                    self.model_name,
                    file_info.mtime,
                    emb_bytes
                )
            )

        # Insert embeddings into vec table for KNN search (skip None/malformed).
        # BUG-10: the guard is PER ROW, not one try around the whole loop. The
        # old blanket except meant the FIRST failing insert (e.g. a width the
        # vec table no longer expects after a dimension change) dropped ALL
        # remaining chunks silently -- a permanently keyword-only index while
        # the operator believed semantic search was live. A per-row failure now
        # skips just that row and is logged; a MissingTable-style error (sqlite-
        # vec not loaded) short-circuits the loop once instead of per row.
        for i, chunk in enumerate(chunks):
            emb_bytes = emb_bytes_list[i]
            if emb_bytes is None:
                continue
            chunk_id = hashlib.sha256(
                f"{file_info.path}:{chunk.start_line}:{i}".encode()
            ).hexdigest()
            try:
                self.db.execute(
                    "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                    (chunk_id, emb_bytes)
                )
            except sqlite3.OperationalError as e:
                # No chunks_vec table at all (sqlite-vec not loaded) -> stop
                # trying for this file; it is keyword-only by configuration.
                if "no such table" in str(e).lower():
                    break
                logger.warning(
                    "Vector insert failed for %s chunk %d: %s",
                    file_info.path, i, e,
                )
            except Exception as e:
                logger.warning(
                    "Vector insert failed for %s chunk %d: %s",
                    file_info.path, i, e,
                )

        self.db.commit()
        return len(chunks)

    def remove_stale(self, current_files: list[FileInfo]) -> int:
        """Remove chunks for files that no longer exist. 
        Compares DB paths against current_files paths.
        Returns number of chunks removed."""
        # Get all unique paths from DB for this model
        cursor = self.db.execute(
            "SELECT DISTINCT path FROM chunks WHERE model = ?",
            (self.model_name,)
        )
        db_paths = {row[0] for row in cursor.fetchall()}
        
        # Build set of current file paths
        current_paths = {f.path for f in current_files}
        
        # Find paths in DB but not in current files
        stale_paths = db_paths - current_paths
        
        # Delete chunks for stale paths
        removed_count = 0
        for stale_path in stale_paths:
            # Clean the vec table too, else embeddings for deleted files linger as
            # orphans (chunks_vec rows with no matching chunk). Collect ids first.
            old_ids = [r[0] for r in self.db.execute(
                "SELECT id FROM chunks WHERE path = ? AND model = ?",
                (stale_path, self.model_name)
            ).fetchall()]
            if old_ids:
                try:
                    placeholders = ",".join("?" for _ in old_ids)
                    self.db.execute(
                        f"DELETE FROM chunks_vec WHERE id IN ({placeholders})",
                        old_ids
                    )
                except Exception:
                    pass  # chunks_vec may not exist if sqlite-vec not loaded
            cursor = self.db.execute(
                "DELETE FROM chunks WHERE path = ? AND model = ?",
                (stale_path, self.model_name)
            )
            removed_count += cursor.rowcount
        
        self.db.commit()
        return removed_count

    async def index_all(self, paths: list[Path]) -> IndexStats:
        """Full indexing pipeline: scan → detect stale → index changed → remove deleted.
        Returns statistics."""
        # Scan for files
        files = self.scan_files(paths)
        files_scanned = len(files)
        
        # Get stale files that need indexing
        stale = self.get_stale_files(files)
        files_indexed = len(stale)
        
        # Index stale files
        chunks_created = 0
        for file_info in stale:
            chunks_created += await self.index_file(file_info)
        
        # Remove chunks for deleted files
        files_removed = self.remove_stale(files)
        
        return IndexStats(
            files_scanned=files_scanned,
            files_indexed=files_indexed,
            chunks_created=chunks_created,
            files_removed=files_removed
        )
