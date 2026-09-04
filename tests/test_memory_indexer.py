"""Tests for the memory indexer: file scanning, change detection, indexing pipeline.

Interface contract:
    MemoryIndexer(db: Connection, embedder: EmbeddingProvider, model_name: str)
    scan_files(paths: list[Path]) -> list[FileInfo]
    get_stale_files(files: list[FileInfo]) -> list[FileInfo]
    index_file(file_info: FileInfo) -> int  # returns chunk count
    index_all(paths: list[Path]) -> IndexStats
    remove_stale(current_files: list[FileInfo]) -> int  # returns removed count

FileInfo: path (str), mtime (float)
IndexStats: files_scanned (int), files_indexed (int), chunks_created (int), files_removed (int)
"""

import pytest
import os
from unittest.mock import MagicMock
from openalph.memory.indexer import MemoryIndexer, FileInfo, IndexStats
from openalph.memory.schema import init_db


def _make_embedder_mock(dims=4):
    """Create a mock EmbeddingProvider that returns fixed-size vectors."""
    embedder = MagicMock()
    embedder.model = "test-model"

    async def mock_embed(text):
        return [0.1] * dims

    async def mock_embed_batch(texts):
        return [[0.1] * dims for _ in texts]

    embedder.embed = mock_embed
    embedder.embed_batch = mock_embed_batch
    return embedder


class TestScanFiles:

    def test_finds_md_files(self, tmp_path):
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "notes.md").write_text("# Notes\nContent here that is long enough.")
        (tmp_path / "memory" / "log.md").write_text("# Log\nAnother file with content.")

        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")
        files = indexer.scan_files([tmp_path / "memory"])
        assert len(files) == 2
        assert all(isinstance(f, FileInfo) for f in files)
        conn.close()

    def test_ignores_non_md_files(self, tmp_path):
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "notes.md").write_text("content long enough")
        (tmp_path / "memory" / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "memory" / "data.json").write_text("{}")

        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")
        files = indexer.scan_files([tmp_path / "memory"])
        assert len(files) == 1
        assert files[0].path.endswith(".md")
        conn.close()

    def test_recurses_subdirectories(self, tmp_path):
        (tmp_path / "memory" / "daily").mkdir(parents=True)
        (tmp_path / "memory" / "daily" / "2026-03-10.md").write_text("daily log")
        (tmp_path / "memory" / "top.md").write_text("top level")

        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")
        files = indexer.scan_files([tmp_path / "memory"])
        assert len(files) == 2
        conn.close()

    def test_handles_individual_files(self, tmp_path):
        """Can pass individual file paths, not just directories."""
        f = tmp_path / "SAFETY.md"
        f.write_text("safety content")

        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")
        files = indexer.scan_files([f])
        assert len(files) == 1
        conn.close()

    def test_skips_missing_paths(self, tmp_path):
        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")
        files = indexer.scan_files([tmp_path / "nonexistent"])
        assert files == []
        conn.close()

    def test_fileinfo_has_mtime(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("content")

        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")
        files = indexer.scan_files([f])
        assert files[0].mtime > 0
        conn.close()


class TestGetStaleFiles:

    def test_new_file_is_stale(self, tmp_path):
        """File not in DB is considered stale (needs indexing)."""
        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")
        files = [FileInfo(path="new.md", mtime=1000.0)]
        stale = indexer.get_stale_files(files)
        assert len(stale) == 1
        conn.close()

    def test_unchanged_file_not_stale(self, tmp_path):
        """File with same mtime as DB is not stale."""
        conn = init_db(tmp_path / "test.db", dimensions=4)
        conn.execute(
            "INSERT INTO chunks (id, path, start_line, end_line, text, source, model, file_mtime) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("c1", "existing.md", 1, 5, "content", "memory", "test-model", 1000.0)
        )
        conn.commit()

        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")
        files = [FileInfo(path="existing.md", mtime=1000.0)]
        stale = indexer.get_stale_files(files)
        assert len(stale) == 0
        conn.close()

    def test_modified_file_is_stale(self, tmp_path):
        """File with newer mtime than DB is stale."""
        conn = init_db(tmp_path / "test.db", dimensions=4)
        conn.execute(
            "INSERT INTO chunks (id, path, start_line, end_line, text, source, model, file_mtime) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("c1", "modified.md", 1, 5, "old content", "memory", "test-model", 1000.0)
        )
        conn.commit()

        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")
        files = [FileInfo(path="modified.md", mtime=2000.0)]
        stale = indexer.get_stale_files(files)
        assert len(stale) == 1
        conn.close()


class TestIndexFile:

    @pytest.mark.asyncio
    async def test_creates_chunks_in_db(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("## Section\nContent that is long enough to meet the minimum chunk size threshold for testing purposes here.")

        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")
        info = FileInfo(path=str(f), mtime=os.path.getmtime(str(f)))
        count = await indexer.index_file(info)

        assert count >= 1
        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        assert cursor.fetchone()[0] >= 1
        conn.close()

    @pytest.mark.asyncio
    async def test_replaces_old_chunks_on_reindex(self, tmp_path):
        """Re-indexing a file replaces its old chunks."""
        f = tmp_path / "test.md"
        f.write_text("## V1\nOriginal content long enough to meet minimum chunk size threshold for testing.")

        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")
        info = FileInfo(path=str(f), mtime=1000.0)
        await indexer.index_file(info)

        # Modify and re-index
        f.write_text("## V2\nUpdated content long enough to meet minimum chunk size threshold for testing purposes.")
        info2 = FileInfo(path=str(f), mtime=2000.0)
        await indexer.index_file(info2)

        cursor = conn.execute("SELECT text FROM chunks WHERE path = ?", (str(f),))
        texts = [row[0] for row in cursor.fetchall()]
        assert any("V2" in t for t in texts)
        assert not any("V1" in t for t in texts)
        conn.close()

    @pytest.mark.asyncio
    async def test_stores_embeddings(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("## Section\nContent that is long enough to meet minimum chunk size for embedding test.")

        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(dims=4), "test-model")
        info = FileInfo(path=str(f), mtime=os.path.getmtime(str(f)))
        await indexer.index_file(info)

        cursor = conn.execute("SELECT embedding FROM chunks WHERE path = ?", (str(f),))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] is not None  # embedding stored
        conn.close()

    @pytest.mark.asyncio
    async def test_handles_embedding_failure(self, tmp_path):
        """If embedder returns None, chunk is still stored (with null embedding)."""
        f = tmp_path / "test.md"
        f.write_text("## Section\nContent that is long enough to meet minimum chunk size for this test.")

        conn = init_db(tmp_path / "test.db", dimensions=4)

        # Embedder that always fails
        embedder = MagicMock()
        embedder.model = "test-model"

        async def mock_embed_batch(texts):
            return [None for _ in texts]

        embedder.embed_batch = mock_embed_batch

        indexer = MemoryIndexer(conn, embedder, "test-model")
        info = FileInfo(path=str(f), mtime=os.path.getmtime(str(f)))
        count = await indexer.index_file(info)

        assert count >= 1
        cursor = conn.execute("SELECT embedding FROM chunks WHERE path = ?", (str(f),))
        row = cursor.fetchone()
        assert row[0] is None  # embedding is null but chunk exists
        conn.close()


class TestRemoveStale:

    def test_removes_chunks_for_deleted_files(self, tmp_path):
        conn = init_db(tmp_path / "test.db", dimensions=4)
        conn.execute(
            "INSERT INTO chunks (id, path, start_line, end_line, text, source, model, file_mtime) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("c1", "deleted.md", 1, 5, "old", "memory", "test-model", 1000.0)
        )
        conn.execute(
            "INSERT INTO chunks (id, path, start_line, end_line, text, source, model, file_mtime) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("c2", "kept.md", 1, 5, "keep", "memory", "test-model", 1000.0)
        )
        conn.commit()

        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")
        current = [FileInfo(path="kept.md", mtime=1000.0)]
        removed = indexer.remove_stale(current)

        assert removed == 1
        cursor = conn.execute("SELECT COUNT(*) FROM chunks WHERE path = ?", ("deleted.md",))
        assert cursor.fetchone()[0] == 0
        cursor = conn.execute("SELECT COUNT(*) FROM chunks WHERE path = ?", ("kept.md",))
        assert cursor.fetchone()[0] == 1
        conn.close()


class TestIndexAll:

    @pytest.mark.asyncio
    async def test_full_pipeline(self, tmp_path):
        """End-to-end: scan, chunk, embed, store."""
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "a.md").write_text(
            "## Topic A\nContent about topic A that is long enough to meet the minimum chunk size threshold."
        )
        (tmp_path / "memory" / "b.md").write_text(
            "## Topic B\nContent about topic B that is long enough to meet the minimum chunk size threshold."
        )

        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")
        stats = await indexer.index_all([tmp_path / "memory"])

        assert isinstance(stats, IndexStats)
        assert stats.files_scanned == 2
        assert stats.files_indexed == 2
        assert stats.chunks_created >= 2
        conn.close()

    @pytest.mark.asyncio
    async def test_incremental_only_indexes_changed(self, tmp_path):
        """Second run only indexes files that changed."""
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "a.md").write_text(
            "## Topic A\nContent about topic A long enough for chunking minimum."
        )
        (tmp_path / "memory" / "b.md").write_text(
            "## Topic B\nContent about topic B long enough for chunking minimum."
        )

        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")

        # First run
        stats1 = await indexer.index_all([tmp_path / "memory"])
        assert stats1.files_indexed == 2

        # Second run (no changes)
        stats2 = await indexer.index_all([tmp_path / "memory"])
        assert stats2.files_indexed == 0  # nothing changed
        conn.close()


class TestEmbeddingDimensionGuard:
    """Indexer rejects malformed (wrong-dimension) embeddings at write time: the
    chunk is still indexed for keyword search but gets no vector, and the
    rejection is logged loudly (regression guard for the silent-corruption bug)."""

    @pytest.mark.asyncio
    async def test_malformed_embedding_skipped_and_logged(self, tmp_path, caplog):
        import logging as _logging

        def _bad_embedder(dims=5):  # vec table is dim 4; 5 is malformed
            embedder = MagicMock()
            embedder.model = "test-model"

            async def mock_embed(text):
                return [0.1] * dims

            async def mock_embed_batch(texts):
                return [[0.1] * dims for _ in texts]

            embedder.embed = mock_embed
            embedder.embed_batch = mock_embed_batch
            return embedder

        f = tmp_path / "test.md"
        f.write_text("## Section\nContent long enough to meet the minimum chunk size threshold for testing purposes here.")

        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _bad_embedder(), "test-model")
        info = FileInfo(path=str(f), mtime=os.path.getmtime(str(f)))

        with caplog.at_level(_logging.WARNING):
            count = await indexer.index_file(info)

        # Chunk is still indexed (keyword-searchable) ...
        assert count >= 1
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] >= 1
        # ... but with NO stored vector (malformed embedding rejected) ...
        non_null = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
        assert non_null == 0
        # ... and the rejection was logged loudly.
        assert "malformed embedding" in caplog.text.lower()
        conn.close()

    @pytest.mark.asyncio
    async def test_valid_embedding_is_stored(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("## Section\nContent long enough to meet the minimum chunk size threshold for testing purposes here.")

        conn = init_db(tmp_path / "test.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(dims=4), "test-model")
        info = FileInfo(path=str(f), mtime=os.path.getmtime(str(f)))
        await indexer.index_file(info)

        non_null = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
        assert non_null >= 1
        conn.close()


# ===========================================================================
# BUG-9 — one unreadable file must not abort the whole index pass
# BUG-10 — a per-row vec-insert guard, not one blanket except over the loop
# ===========================================================================

class TestIndexerRobustness:

    @pytest.mark.asyncio
    async def test_unreadable_file_is_skipped_not_fatal(self, tmp_path):
        """BUG-9: read_text had no error handling, so a non-UTF-8/deleted file
        raised out of index_all and aborted indexing for every other file."""
        (tmp_path / "good.md").write_text("## Good\n\n" + "plenty of indexable content in this file. " * 3 + "\n")
        (tmp_path / "bad.md").write_bytes(b"## Bad\n\n\xff\xfe not utf-8 \xff\n")

        conn = init_db(tmp_path / "idx.db", dimensions=4)
        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")

        stats = await indexer.index_all([tmp_path])  # must not raise

        paths = [row[0] for row in conn.execute("SELECT DISTINCT path FROM chunks")]
        assert any("good.md" in p for p in paths)
        assert not any("bad.md" in p for p in paths)

    @pytest.mark.asyncio
    async def test_indexes_when_vec_table_absent(self, tmp_path):
        """BUG-10: without sqlite-vec the chunks_vec table doesn't exist; the
        per-row guard must skip vector inserts while still indexing for FTS."""
        (tmp_path / "a.md").write_text("## A\n\n" + "searchable keyword content in this doc. " * 3 + "\n")
        conn = init_db(tmp_path / "idx.db", dimensions=4)  # no load_vec_extension
        indexer = MemoryIndexer(conn, _make_embedder_mock(), "test-model")

        stats = await indexer.index_all([tmp_path])  # must not raise

        n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert n >= 1
