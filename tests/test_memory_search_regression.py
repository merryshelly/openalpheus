"""Regression tests for two memory-search bug fixes.

FIX 1 — memory_search.py / run_memory_search:
    vector-search result loop previously crashed (TypeError: 1.0 - None) when
    vec_distance_cosine returned NULL for a malformed/NULL-embedding row, silently
    zeroing ALL vector results. Fix: skip rows where dist is None.

FIX 2 — indexer.py / MemoryIndexer.remove_stale:
    previously deleted rows from chunks but left chunks_vec rows for the same IDs
    (orphans). Fix: collect chunk ids for stale paths, DELETE from chunks_vec
    before deleting from chunks.

Style follows test_memory_indexer.py: init_db + load_vec_extension, fake embedder
with async embed/embed_batch, FileInfo fixture pattern.
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from openalph.memory.indexer import FileInfo, IndexStats, MemoryIndexer
from openalph.memory.schema import init_db, load_vec_extension


# ---------------------------------------------------------------------------
# Shared helpers — mirror _make_embedder_mock from test_memory_indexer.py
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FIX 2 regression: remove_stale must clean up chunks_vec (no orphans)
# ---------------------------------------------------------------------------

class TestRemoveStaleVecOrphans:
    """Regression for FIX 2: remove_stale must delete matching chunks_vec rows."""

    @pytest.mark.asyncio
    async def test_remove_stale_cleans_chunks_vec(self, tmp_path):
        """After remove_stale, no chunks_vec row should be left without a matching
        chunks row. Previously remove_stale deleted from chunks but not chunks_vec,
        leaving orphaned vector embeddings."""
        # Set up DB with sqlite-vec extension
        conn = init_db(tmp_path / "test.db", dimensions=4)
        vec_loaded = load_vec_extension(conn)
        if not vec_loaded:
            pytest.skip("sqlite-vec extension not available")

        # Create a real file so index_file has content to chunk
        f = tmp_path / "stale_file.md"
        f.write_text(
            "## Section\n"
            "Content that is long enough to meet the minimum chunk size threshold for testing purposes."
        )

        indexer = MemoryIndexer(conn, _make_embedder_mock(dims=4), "test-model")
        info = FileInfo(path=str(f), mtime=os.path.getmtime(str(f)))

        # Index the file → creates rows in both chunks and chunks_vec
        chunk_count = await indexer.index_file(info)
        assert chunk_count >= 1, "Expected at least one chunk to be created"

        # Capture the chunk IDs that were indexed
        stale_chunk_ids = {
            row[0]
            for row in conn.execute(
                "SELECT id FROM chunks WHERE path = ?", (str(f),)
            ).fetchall()
        }
        assert stale_chunk_ids, "Expected chunk IDs in DB after index_file"

        # Verify chunks_vec rows exist for those IDs
        vec_ids_before = {
            row[0] for row in conn.execute("SELECT id FROM chunks_vec").fetchall()
        }
        assert stale_chunk_ids.issubset(vec_ids_before), (
            "Expected chunks_vec rows for every indexed chunk before remove_stale"
        )

        # Now call remove_stale with empty current_files → file is stale
        removed = indexer.remove_stale([])
        assert removed == chunk_count, (
            f"remove_stale should have removed {chunk_count} chunk(s), got {removed}"
        )

        # KEY ASSERTION (regression): every chunks_vec.id must still have a
        # matching chunks.id. No orphaned vector rows may remain.
        remaining_chunk_ids = {
            row[0] for row in conn.execute("SELECT id FROM chunks").fetchall()
        }
        remaining_vec_ids = {
            row[0] for row in conn.execute("SELECT id FROM chunks_vec").fetchall()
        }
        orphaned = remaining_vec_ids - remaining_chunk_ids
        assert not orphaned, (
            f"FIX 2 regression: remove_stale left {len(orphaned)} orphaned "
            f"chunks_vec row(s): {orphaned}"
        )

        conn.close()

    @pytest.mark.asyncio
    async def test_remove_stale_preserves_live_vec_rows(self, tmp_path):
        """remove_stale must only delete chunks_vec rows for the stale file,
        leaving the live file's vector rows intact."""
        conn = init_db(tmp_path / "test.db", dimensions=4)
        vec_loaded = load_vec_extension(conn)
        if not vec_loaded:
            pytest.skip("sqlite-vec extension not available")

        # Index two files
        stale_f = tmp_path / "stale.md"
        stale_f.write_text(
            "## Stale\nThis file will be removed. Content is long enough for chunking."
        )
        live_f = tmp_path / "live.md"
        live_f.write_text(
            "## Live\nThis file stays. Content is long enough to meet chunk size threshold."
        )

        indexer = MemoryIndexer(conn, _make_embedder_mock(dims=4), "test-model")

        stale_info = FileInfo(path=str(stale_f), mtime=os.path.getmtime(str(stale_f)))
        live_info = FileInfo(path=str(live_f), mtime=os.path.getmtime(str(live_f)))

        await indexer.index_file(stale_info)
        await indexer.index_file(live_info)

        live_chunk_ids = {
            row[0]
            for row in conn.execute(
                "SELECT id FROM chunks WHERE path = ?", (str(live_f),)
            ).fetchall()
        }
        assert live_chunk_ids

        # Remove stale, keeping live file current
        removed = indexer.remove_stale([live_info])
        assert removed >= 1

        # Live file's chunks_vec rows must still be present
        remaining_vec_ids = {
            row[0] for row in conn.execute("SELECT id FROM chunks_vec").fetchall()
        }
        missing_live_vecs = live_chunk_ids - remaining_vec_ids
        assert not missing_live_vecs, (
            f"remove_stale incorrectly deleted live chunks_vec rows: {missing_live_vecs}"
        )

        # And no orphans for the stale file
        remaining_chunk_ids = {
            row[0] for row in conn.execute("SELECT id FROM chunks").fetchall()
        }
        orphaned = remaining_vec_ids - remaining_chunk_ids
        assert not orphaned, (
            f"FIX 2 regression: orphaned chunks_vec rows after remove_stale: {orphaned}"
        )

        conn.close()


# ---------------------------------------------------------------------------
# FIX 1 regression: NULL cosine-distance must not crash vector-search loop
# ---------------------------------------------------------------------------

class TestVectorSearchNullDistanceGuard:
    """Regression for FIX 1: NULL dist from vec_distance_cosine must be skipped,
    not crash the loop with TypeError(1.0 - None)."""

    @pytest.mark.asyncio
    async def test_null_dist_row_does_not_crash_search(self, tmp_path):
        """When the vector-search query returns a row with NULL cosine distance
        (row[6] is None), run_memory_search must NOT raise and must still return
        the valid (non-NULL) results."""
        from openalph.tools.memory_search import _index_cache, run_memory_search

        workspace_key = str(tmp_path)

        # Workspace must exist and have at least a memory dir so scan works
        (tmp_path / "memory").mkdir()

        # Build a mock MemoryIndexer whose db.execute returns:
        #   - empty FTS results
        #   - one NULL-dist row followed by one valid row for the vector query
        null_row = (
            "chunk_null", "null_embedding_file.md", "memory", 1, 5,
            "content from a chunk whose distance is NULL", None  # row[6] = NULL dist
        )
        good_row = (
            "chunk_good", "good_embedding_file.md", "memory", 1, 5,
            "content from a well-embedded chunk that should be returned", 0.15
        )

        def fake_execute(sql, params=None):
            cursor = MagicMock()
            sql_upper = sql.strip().upper()
            if "MATCH" in sql_upper:           # BM25 / FTS query
                cursor.fetchall.return_value = []
            elif "VEC_DISTANCE_COSINE" in sql_upper:  # vector query
                cursor.fetchall.return_value = [null_row, good_row]
            else:
                cursor.fetchall.return_value = []
            return cursor

        mock_db = MagicMock()
        mock_db.execute = fake_execute

        mock_indexer = MagicMock()
        mock_indexer.model_name = "test-model"
        mock_indexer.db = mock_db
        # Embedder returns a 4-dim vector so query_blob is valid
        mock_indexer.embedder = MagicMock()
        mock_indexer.embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        # index_all is a no-op (nothing to re-index)
        mock_indexer.index_all = AsyncMock(
            return_value=IndexStats(files_scanned=0, files_indexed=0,
                                   chunks_created=0, files_removed=0)
        )

        # Inject mock indexer so run_memory_search reuses it
        _index_cache[workspace_key] = mock_indexer
        try:
            config = {
                "embedding_model": "test",
                "embedding_base_url": "http://localhost",
                "vector_weight": 0.7,
                "text_weight": 0.3,
                "mmr_enabled": False,
                "temporal_decay_enabled": False,
            }

            # This MUST NOT raise TypeError (the pre-fix crash: 1.0 - None)
            result = await run_memory_search(
                query="some test query", config=config, workspace=tmp_path
            )

            # Must not be an error result
            assert not result.is_error, (
                f"FIX 1 regression: run_memory_search returned error: {result.content}"
            )

            # The valid (non-NULL) row must appear in results
            assert "good_embedding_file.md" in result.content, (
                "Expected the good (non-NULL-dist) result to be present in output"
            )

            # The NULL-dist row must NOT appear (it was skipped)
            assert "null_embedding_file.md" not in result.content, (
                "NULL-dist row must be skipped, not appear in search results"
            )
        finally:
            _index_cache.pop(workspace_key, None)

    @pytest.mark.asyncio
    async def test_all_null_dist_rows_returns_no_results(self, tmp_path):
        """If ALL vector rows have NULL dist (all malformed embeddings), the search
        should gracefully return 'no results' rather than crashing."""
        from openalph.tools.memory_search import _index_cache, run_memory_search

        workspace_key = str(tmp_path)
        (tmp_path / "memory").mkdir()

        # ALL rows have NULL dist
        null_row_1 = ("c1", "file1.md", "memory", 1, 5, "content one", None)
        null_row_2 = ("c2", "file2.md", "memory", 1, 5, "content two", None)

        def fake_execute(sql, params=None):
            cursor = MagicMock()
            sql_upper = sql.strip().upper()
            if "MATCH" in sql_upper:
                cursor.fetchall.return_value = []
            elif "VEC_DISTANCE_COSINE" in sql_upper:
                cursor.fetchall.return_value = [null_row_1, null_row_2]
            else:
                cursor.fetchall.return_value = []
            return cursor

        mock_db = MagicMock()
        mock_db.execute = fake_execute

        mock_indexer = MagicMock()
        mock_indexer.model_name = "test-model"
        mock_indexer.db = mock_db
        mock_indexer.embedder = MagicMock()
        mock_indexer.embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        mock_indexer.index_all = AsyncMock(
            return_value=IndexStats(
                files_scanned=0, files_indexed=0, chunks_created=0, files_removed=0
            )
        )

        _index_cache[workspace_key] = mock_indexer
        try:
            config = {
                "embedding_model": "test",
                "embedding_base_url": "http://localhost",
                "vector_weight": 0.7,
                "text_weight": 0.3,
                "mmr_enabled": False,
                "temporal_decay_enabled": False,
            }

            # Must not raise; should return a graceful "no results" message
            result = await run_memory_search(
                query="some test query", config=config, workspace=tmp_path
            )

            assert not result.is_error, (
                f"FIX 1 regression: unexpected error when all dists are NULL: {result.content}"
            )
            # No crash = fix is working; content is "no results"
            assert "no results" in result.content.lower() or result.content, (
                "Expected a non-empty, non-error response"
            )
        finally:
            _index_cache.pop(workspace_key, None)
