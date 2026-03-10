"""Memory search tool executor.

Thin wrapper around MemoryIndex that handles tool config,
workspace resolution, and output formatting.
"""

import logging
import re
from pathlib import Path

from openalph.memory.schema import init_db, load_vec_extension
from openalph.memory.embeddings import EmbeddingProvider
from openalph.memory.indexer import MemoryIndexer
from openalph.memory.search import (
    build_fts_query,
    bm25_rank_to_score,
    merge_hybrid_results,
    apply_temporal_decay,
    mmr_rerank,
    SearchResult,
)
from openalph.tools import ToolResult

logger = logging.getLogger(__name__)

# Cache MemoryIndexer per workspace to avoid re-loading the model on every call
_index_cache: dict[str, "MemoryIndexer"] = {}


def _get_scan_paths(workspace: Path, extra_paths: list[str] | None = None) -> list[Path]:
    """Build the list of paths to scan for memory files."""
    paths = []
    # Default: memory/ directory
    memory_dir = workspace / "memory"
    if memory_dir.exists():
        paths.append(memory_dir)
    # Root prompt files
    for name in ("SAFETY.md", "SOUL.md", "OPERATOR.md", "OPERATIONS.md",
                 "ENVIRONMENT.md", "WAKE.md"):
        f = workspace / name
        if f.exists():
            paths.append(f)
    # Skills
    skills_dir = workspace / "skills"
    if skills_dir.exists():
        paths.append(skills_dir)
    # Extra paths from config
    if extra_paths:
        for p in extra_paths:
            pp = Path(p)
            if pp.exists():
                paths.append(pp)
    return paths


async def run_memory_search(
    query: str,
    config: dict,
    workspace: Path,
    max_results: int = 10,
    min_score: float = 0.1,
) -> ToolResult:
    """Execute a memory search query against the workspace.

    Returns formatted results with path:line citations and snippets.
    """
    if not query or not query.strip():
        return ToolResult(content="Error: empty search query", is_error=True)

    workspace = Path(workspace)
    workspace_key = str(workspace)

    # Initialize or reuse indexer
    if workspace_key not in _index_cache:
        db_dir = workspace / ".memory-index"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / "search.db"

        model_path = config.get("embedding_model", "/opt/openalph/models/nomic-embed-text-v1.5.Q8_0.gguf")
        base_url = config.get("embedding_base_url", "http://localhost:11434")

        embedder = EmbeddingProvider(model=model_path, base_url=base_url)
        dimensions = 768  # nomic-embed-text default

        conn = init_db(db_path, dimensions=dimensions)
        vec_loaded = load_vec_extension(conn)
        if not vec_loaded:
            logger.warning("sqlite-vec not available — vector search disabled, using BM25 only")

        indexer = MemoryIndexer(conn, embedder, model_name="nomic-embed-text-v1.5")
        _index_cache[workspace_key] = indexer

    indexer = _index_cache[workspace_key]

    # Ensure index is up to date
    scan_paths = _get_scan_paths(workspace, config.get("extra_paths"))
    stats = await indexer.index_all(scan_paths)
    if stats.files_indexed > 0:
        logger.info("Indexed %d files (%d chunks)", stats.files_indexed, stats.chunks_created)

    # Search
    vector_weight = config.get("vector_weight", 0.7)
    text_weight = config.get("text_weight", 0.3)
    mmr_enabled = config.get("mmr_enabled", True)
    mmr_lambda = config.get("mmr_lambda", 0.7)
    decay_enabled = config.get("temporal_decay_enabled", True)
    decay_half_life = config.get("temporal_decay_half_life_days", 30)

    db = indexer.db
    model_name = indexer.model_name

    # BM25 keyword search
    keyword_results = []
    fts_query = build_fts_query(query)
    if fts_query:
        try:
            rows = db.execute(
                "SELECT id, path, source, start_line, end_line, text, "
                "bm25(chunks_fts) AS rank "
                "FROM chunks_fts WHERE chunks_fts MATCH ? AND model = ? "
                "ORDER BY rank ASC LIMIT ?",
                (fts_query, model_name, max_results * 3),
            ).fetchall()
            for row in rows:
                keyword_results.append({
                    "id": row[0], "path": row[1], "source": row[2],
                    "start_line": row[3], "end_line": row[4],
                    "snippet": row[5][:500],
                    "text_score": bm25_rank_to_score(row[6]),
                })
        except Exception as e:
            logger.warning("FTS search failed: %s", e)

    # Vector search
    vector_results = []
    query_embedding = await indexer.embedder.embed(query)
    if query_embedding is not None:
        try:
            import struct
            query_blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)
            # Try sqlite-vec KNN search
            rows = db.execute(
                "SELECT v.id, c.path, c.source, c.start_line, c.end_line, c.text, "
                "vec_distance_cosine(v.embedding, ?) AS dist "
                "FROM chunks_vec v JOIN chunks c ON c.id = v.id "
                "WHERE c.model = ? "
                "ORDER BY dist ASC LIMIT ?",
                (query_blob, model_name, max_results * 3),
            ).fetchall()
            for row in rows:
                vector_results.append({
                    "id": row[0], "path": row[1], "source": row[2],
                    "start_line": row[3], "end_line": row[4],
                    "snippet": row[5][:500],
                    "vector_score": max(0, 1.0 - row[6]),
                })
        except Exception as e:
            logger.debug("Vector search unavailable: %s", e)

    # If no vector results, use text_weight=1.0
    if not vector_results:
        text_weight = 1.0
        vector_weight = 0.0

    # Merge
    results = merge_hybrid_results(
        vector=vector_results,
        keyword=keyword_results,
        vector_weight=vector_weight,
        text_weight=text_weight,
    )

    # Temporal decay
    if decay_enabled:
        import time
        results = apply_temporal_decay(
            results,
            half_life_days=decay_half_life,
            now_ms=int(time.time() * 1000),
        )

    # MMR re-ranking
    if mmr_enabled and len(results) > 1:
        results = mmr_rerank(results, lambda_param=mmr_lambda)

    # Filter by min_score and limit
    results = [r for r in results if r.score >= min_score][:max_results]

    # Format output
    if not results:
        return ToolResult(content=f"No results found for \"{query}\".", is_error=False)

    lines = [f"Found {len(results)} results for \"{query}\":\n"]
    for i, r in enumerate(results, 1):
        snippet = r.snippet[:200].replace("\n", " ").strip()
        lines.append(f"[{i}] {r.path}:{r.start_line}-{r.end_line} (score: {r.score:.2f})")
        lines.append(f"  {snippet}\n")

    return ToolResult(content="\n".join(lines), is_error=False)
