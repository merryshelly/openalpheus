"""Tests for hybrid search pipeline: BM25 + vector, temporal decay, MMR.

Interface contract:
    build_fts_query(raw: str) -> str | None
    bm25_rank_to_score(rank: float) -> float
    merge_hybrid_results(vector, keyword, vector_weight, text_weight, ...) -> list[SearchResult]
    apply_temporal_decay(results, half_life_days, now_ms) -> list[SearchResult]
    mmr_rerank(results, lambda_param) -> list[SearchResult]

SearchResult: path (str), start_line (int), end_line (int), score (float),
             snippet (str), source (str)
"""

import pytest
from openalph.memory.search import (
    build_fts_query,
    bm25_rank_to_score,
    merge_hybrid_results,
    apply_temporal_decay,
    mmr_rerank,
    SearchResult,
)


class TestBuildFtsQuery:

    def test_simple_query(self):
        result = build_fts_query("validator monitoring")
        assert result is not None
        assert "validator" in result.lower()
        assert "monitoring" in result.lower()

    def test_empty_string_returns_none(self):
        assert build_fts_query("") is None

    def test_whitespace_only_returns_none(self):
        assert build_fts_query("   ") is None

    def test_special_chars_stripped(self):
        """FTS special characters should be handled without error."""
        result = build_fts_query("hello! @world #test")
        assert result is not None
        assert "hello" in result.lower()

    def test_quotes_escaped(self):
        """Quotes in input should not break FTS syntax."""
        result = build_fts_query('say "hello" world')
        assert result is not None


class TestBm25RankToScore:

    def test_zero_rank_gives_high_score(self):
        """Rank 0 (best match) → score near 1.0."""
        score = bm25_rank_to_score(0.0)
        assert score > 0.9

    def test_high_rank_gives_low_score(self):
        """High rank (poor match) → score near 0."""
        score = bm25_rank_to_score(100.0)
        assert score < 0.1

    def test_negative_rank_treated_as_zero(self):
        """BM25 can return negative rank values; clamp to 0."""
        score = bm25_rank_to_score(-5.0)
        assert score > 0.9

    def test_score_always_between_0_and_1(self):
        for rank in [-10, 0, 0.5, 1, 5, 50, 999]:
            score = bm25_rank_to_score(rank)
            assert 0 <= score <= 1


class TestMergeHybridResults:

    def test_vector_only(self):
        """When only vector results exist, they are returned scored."""
        vector = [
            {"id": "a", "path": "a.md", "start_line": 1, "end_line": 5,
             "snippet": "hello", "source": "memory", "vector_score": 0.9},
        ]
        results = merge_hybrid_results(
            vector=vector, keyword=[], vector_weight=0.7, text_weight=0.3
        )
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.7 * 0.9)

    def test_keyword_only(self):
        keyword = [
            {"id": "b", "path": "b.md", "start_line": 1, "end_line": 3,
             "snippet": "world", "source": "memory", "text_score": 0.8},
        ]
        results = merge_hybrid_results(
            vector=[], keyword=keyword, vector_weight=0.7, text_weight=0.3
        )
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.3 * 0.8)

    def test_overlapping_results_merged(self):
        """Same chunk ID in both vector and keyword → scores combined."""
        vector = [
            {"id": "c", "path": "c.md", "start_line": 1, "end_line": 5,
             "snippet": "test", "source": "memory", "vector_score": 0.8},
        ]
        keyword = [
            {"id": "c", "path": "c.md", "start_line": 1, "end_line": 5,
             "snippet": "test", "source": "memory", "text_score": 0.6},
        ]
        results = merge_hybrid_results(
            vector=vector, keyword=keyword, vector_weight=0.7, text_weight=0.3
        )
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.7 * 0.8 + 0.3 * 0.6)

    def test_results_sorted_by_score_descending(self):
        vector = [
            {"id": "low", "path": "low.md", "start_line": 1, "end_line": 1,
             "snippet": "low", "source": "memory", "vector_score": 0.3},
            {"id": "high", "path": "high.md", "start_line": 1, "end_line": 1,
             "snippet": "high", "source": "memory", "vector_score": 0.9},
        ]
        results = merge_hybrid_results(
            vector=vector, keyword=[], vector_weight=1.0, text_weight=0.0
        )
        assert results[0].path == "high.md"
        assert results[1].path == "low.md"

    def test_empty_inputs(self):
        results = merge_hybrid_results(
            vector=[], keyword=[], vector_weight=0.7, text_weight=0.3
        )
        assert results == []


class TestTemporalDecay:

    def test_recent_file_no_decay(self):
        """Files from today should have score ~unchanged."""
        results = [
            SearchResult(path="memory/2026-03-10.md", start_line=1, end_line=5,
                        score=1.0, snippet="today", source="memory"),
        ]
        decayed = apply_temporal_decay(
            results, half_life_days=30, now_ms=1773100800000  # 2026-03-10
        )
        assert decayed[0].score > 0.95

    def test_old_file_decayed(self):
        """Files from 60 days ago should lose significant score."""
        results = [
            SearchResult(path="memory/2026-01-09.md", start_line=1, end_line=5,
                        score=1.0, snippet="old", source="memory"),
        ]
        decayed = apply_temporal_decay(
            results, half_life_days=30, now_ms=1773100800000  # 2026-03-10
        )
        # 60 days / 30 day half-life = 2 half-lives → score ≈ 0.25
        assert decayed[0].score < 0.35

    def test_evergreen_files_not_decayed(self):
        """Non-dated files in memory/ are evergreen and not decayed."""
        results = [
            SearchResult(path="memory/NETWORK.md", start_line=1, end_line=5,
                        score=1.0, snippet="evergreen", source="memory"),
        ]
        decayed = apply_temporal_decay(
            results, half_life_days=30, now_ms=1773100800000
        )
        assert decayed[0].score == 1.0

    def test_memory_md_root_not_decayed(self):
        """MEMORY.md (root) is evergreen."""
        results = [
            SearchResult(path="MEMORY.md", start_line=1, end_line=5,
                        score=1.0, snippet="root", source="memory"),
        ]
        decayed = apply_temporal_decay(
            results, half_life_days=30, now_ms=1773100800000
        )
        assert decayed[0].score == 1.0

    def test_dated_path_regex(self):
        """Only memory/YYYY-MM-DD.md paths are treated as dated."""
        results = [
            SearchResult(path="memory/projects/README.md", start_line=1, end_line=5,
                        score=1.0, snippet="project", source="memory"),
        ]
        decayed = apply_temporal_decay(
            results, half_life_days=30, now_ms=1773100800000
        )
        # Not dated → evergreen → no decay
        assert decayed[0].score == 1.0


class TestMMRRerank:

    def test_preserves_all_results(self):
        results = [
            SearchResult(path="a.md", start_line=1, end_line=1,
                        score=0.9, snippet="unique content alpha", source="memory"),
            SearchResult(path="b.md", start_line=1, end_line=1,
                        score=0.8, snippet="unique content beta", source="memory"),
        ]
        reranked = mmr_rerank(results, lambda_param=0.7)
        assert len(reranked) == 2

    def test_diverse_results_maintained(self):
        """Diverse results should keep their order (no penalty)."""
        results = [
            SearchResult(path="a.md", start_line=1, end_line=1,
                        score=0.9, snippet="validator monitoring alerts", source="memory"),
            SearchResult(path="b.md", start_line=1, end_line=1,
                        score=0.8, snippet="financial portfolio tracking", source="memory"),
        ]
        reranked = mmr_rerank(results, lambda_param=0.7)
        # First result should still be highest scored (diverse, no penalty)
        assert reranked[0].path == "a.md"

    def test_duplicate_content_penalized(self):
        """Near-identical snippets should get demoted."""
        results = [
            SearchResult(path="a.md", start_line=1, end_line=1,
                        score=0.9, snippet="validator monitoring alerts system", source="memory"),
            SearchResult(path="b.md", start_line=1, end_line=1,
                        score=0.85, snippet="validator monitoring alerts config", source="memory"),
            SearchResult(path="c.md", start_line=1, end_line=1,
                        score=0.7, snippet="financial portfolio tracking system", source="memory"),
        ]
        reranked = mmr_rerank(results, lambda_param=0.7)
        # The diverse result (c) should be promoted above the near-duplicate (b)
        paths = [r.path for r in reranked]
        assert paths.index("c.md") < paths.index("b.md")

    def test_lambda_one_is_pure_relevance(self):
        """λ=1.0 means no diversity penalty → original order preserved."""
        results = [
            SearchResult(path="a.md", start_line=1, end_line=1,
                        score=0.9, snippet="same same same", source="memory"),
            SearchResult(path="b.md", start_line=1, end_line=1,
                        score=0.8, snippet="same same same", source="memory"),
        ]
        reranked = mmr_rerank(results, lambda_param=1.0)
        assert reranked[0].path == "a.md"
        assert reranked[1].path == "b.md"

    def test_single_result(self):
        results = [
            SearchResult(path="a.md", start_line=1, end_line=1,
                        score=0.9, snippet="only one", source="memory"),
        ]
        reranked = mmr_rerank(results, lambda_param=0.7)
        assert len(reranked) == 1

    def test_empty_results(self):
        reranked = mmr_rerank([], lambda_param=0.7)
        assert reranked == []
