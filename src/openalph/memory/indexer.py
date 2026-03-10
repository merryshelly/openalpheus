"""File scanner, change detector, and indexing pipeline for memory search.

Ties together the chunker, embeddings, and schema modules.
"""

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from openalph.memory.chunker import chunk_file


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
        # Read file content
        text = Path(file_info.path).read_text()
        
        # Chunk the file
        chunks = chunk_file(text, file_info.path)
        
        if not chunks:
            return 0
        
        # Embed all chunk texts
        texts = [c.text for c in chunks]
        embeddings = await self.embedder.embed_batch(texts)
        
        # Delete old chunks for this file
        # Also delete from vec table if it exists
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
        
        # Insert new chunks
        for i, chunk in enumerate(chunks):
            # Generate ID: hash of path:start_line
            chunk_id = hashlib.sha256(
                f"{file_info.path}:{chunk.start_line}:{i}".encode()
            ).hexdigest()
            
            # Get embedding for this chunk (may be None)
            emb = embeddings[i] if i < len(embeddings) else None
            
            # Convert embedding to bytes if present
            if emb is not None:
                emb_bytes = struct.pack(f'{len(emb)}f', *emb)
            else:
                emb_bytes = None
            
            # Insert chunk
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
        
        # Insert embeddings into vec table for KNN search
        try:
            for i, chunk in enumerate(chunks):
                emb = embeddings[i] if i < len(embeddings) else None
                if emb is not None:
                    chunk_id = hashlib.sha256(
                        f"{file_info.path}:{chunk.start_line}:{i}".encode()
                    ).hexdigest()
                    emb_bytes = struct.pack(f"{len(emb)}f", *emb)
                    self.db.execute(
                        "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                        (chunk_id, emb_bytes)
                    )
        except Exception:
            pass  # chunks_vec may not exist if sqlite-vec not loaded
        
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
