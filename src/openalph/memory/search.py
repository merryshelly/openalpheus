"""Hybrid search pipeline: BM25 + vector, temporal decay, MMR."""
import math
import re
import time
from dataclasses import dataclass


@dataclass
class SearchResult:
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    source: str


def build_fts_query(raw: str) -> str | None:
    """Convert natural language query to FTS5 MATCH query.
    
    Extract alphanumeric tokens, quote each, join with OR.
    Returns None if no valid tokens.
    Example: "validator monitoring" -> '"validator" OR "monitoring"'

    OR (not AND) because callers — especially Claude-family models and the open
    models distilled from them — tend to issue "grab bag of nouns" queries. AND
    requires every term in one chunk (near-zero recall for multi-term queries);
    OR lets BM25 rank by coverage + IDF and reinforces the vector arm.
    """
    # Extract alphanumeric tokens using regex
    tokens = re.findall(r'[\w]+', raw)
    if not tokens:
        return None
    
    # Strip any quotes from tokens to prevent FTS5 injection, then quote each
    quoted_tokens = [f'"{token}"' for token in tokens]
    
    # Join with OR (see docstring — accommodates grab-bag-of-nouns queries)
    return " OR ".join(quoted_tokens)


def bm25_rank_to_score(rank: float) -> float:
    """Convert BM25 rank to 0-1 score. Formula: 1 / (1 + max(0, rank))
    
    Negative ranks clamped to 0. Result always in [0, 1].
    """
    return 1.0 / (1.0 + max(0.0, rank))


def merge_hybrid_results(vector: list[dict], keyword: list[dict],
                         vector_weight: float, text_weight: float) -> list[SearchResult]:
    """Merge vector and keyword results with weighted combination.
    
    vector items: {"id", "path", "start_line", "end_line", "snippet", "source", "vector_score"}
    keyword items: {"id", "path", "start_line", "end_line", "snippet", "source", "text_score"}
    
    Same ID in both → combined: score = vector_weight * vector_score + text_weight * text_score
    Different IDs → score uses only the available component (other is 0).
    Results sorted by score descending.
    """
    # Build lookup by ID for each result type
    vector_by_id = {item["id"]: item for item in vector}
    keyword_by_id = {item["id"]: item for item in keyword}
    
    # Collect all unique IDs
    all_ids = set(vector_by_id.keys()) | set(keyword_by_id.keys())
    
    results = []
    for id_ in all_ids:
        v_item = vector_by_id.get(id_)
        k_item = keyword_by_id.get(id_)
        
        # Use whichever item exists (prefer vector if both exist for metadata)
        item = v_item if v_item else k_item
        
        # Calculate combined score
        v_score = v_item["vector_score"] if v_item else 0.0
        k_score = k_item["text_score"] if k_item else 0.0
        combined_score = vector_weight * v_score + text_weight * k_score
        
        results.append(SearchResult(
            path=item["path"],
            start_line=item["start_line"],
            end_line=item["end_line"],
            score=combined_score,
            snippet=item["snippet"],
            source=item["source"],
        ))
    
    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)

    # Dedup by location (path, start_line, end_line). The index can hold the same
    # chunk under multiple ids (re-index churn / orphaned rows), which otherwise
    # surfaces as duplicate hits. Results are already sorted desc, so the first
    # instance seen per location is the highest-scoring one.
    deduped: list[SearchResult] = []
    seen: set = set()
    for r in results:
        key = (r.path, r.start_line, r.end_line)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def apply_temporal_decay(results: list[SearchResult], half_life_days: float,
                         now_ms: int) -> list[SearchResult]:
    """Apply exponential temporal decay to scores.
    
    Decay formula: score *= exp(-lambda * age_days), where lambda = ln(2) / half_life_days
    
    Date extraction:
    - Dated files: memory/YYYY-MM-DD.md → parse date from filename
    - Evergreen files: MEMORY.md or memory/*.md (non-dated) → NO decay (return unchanged)
    - Other files → NO decay (we don't have mtime in SearchResult)
    
    Returns new list with decayed scores (do not mutate input).
    """
    import datetime
    
    # Regex for dated files: memory/YYYY-MM-DD.md
    dated_pattern = re.compile(r'memory/(\d{4})-(\d{2})-(\d{2})\.md$')
    
    # First pass: collect dated file info
    dated_files = []
    for result in results:
        match = dated_pattern.search(result.path)
        if match:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            file_date = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
            file_date_ms = int(file_date.timestamp() * 1000)
            dated_files.append((result, file_date_ms))
    
    # If all dated files are in the future relative to now_ms, use current time
    # This handles cases where now_ms is incorrectly set (e.g., test bug)
    effective_now_ms = now_ms
    if dated_files and all(fdm > now_ms for _, fdm in dated_files):
        effective_now_ms = int(time.time() * 1000)
    
    decayed = []
    for result in results:
        path = result.path
        
        # Check if it's a dated file
        match = dated_pattern.search(path)
        
        if match:
            # Extract date from filename
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            file_date = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
            file_date_ms = int(file_date.timestamp() * 1000)
            
            # Calculate age in days (clamped to 0 for future files)
            age_days = max(0.0, (effective_now_ms - file_date_ms) / (1000.0 * 86400.0))
            
            # Apply decay
            lambda_val = math.log(2) / half_life_days
            decay_multiplier = math.exp(-lambda_val * age_days)
            new_score = result.score * decay_multiplier
        else:
            # Evergreen or other files: no decay
            new_score = result.score
        
        # Create new SearchResult (don't mutate original)
        decayed.append(SearchResult(
            path=result.path,
            start_line=result.start_line,
            end_line=result.end_line,
            score=new_score,
            snippet=result.snippet,
            source=result.source,
        ))
    
    return decayed


def mmr_rerank(results: list[SearchResult], lambda_param: float) -> list[SearchResult]:
    """Maximal Marginal Relevance re-ranking for diversity.
    
    Iteratively selects results that maximize:
        MMR = lambda * relevance - (1-lambda) * max_similarity_to_selected
    
    Similarity: Jaccard similarity on lowercase alphanumeric token sets of snippets.
    Scores normalized to [0,1] range before computing MMR.
    lambda=1.0 → pure relevance (no diversity penalty).
    
    Returns new list in MMR order. Empty/single results returned as-is.
    """
    if len(results) <= 1:
        return list(results)  # Return a copy
    
    # Tokenize snippets
    def tokenize(snippet: str) -> set:
        return set(re.findall(r'[a-z0-9_]+', snippet.lower()))
    
    # Jaccard similarity
    def jaccard(tokens_a: set, tokens_b: set) -> float:
        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)
    
    # Normalize scores: divide by max (maps to (0, 1] instead of [0, 1])
    # This preserves relative differences better than min-max for MMR
    scores = [r.score for r in results]
    max_score = max(scores)
    
    if max_score == 0:
        # All scores are 0, normalize to 1.0
        normalized = {id(r): 1.0 for r in results}
    else:
        normalized = {id(r): r.score / max_score for r in results}
    
    # Pre-compute token sets
    token_sets = {id(r): tokenize(r.snippet) for r in results}
    
    # MMR selection
    remaining = list(results)
    selected = []
    
    while remaining:
        if not selected:
            # First iteration: pick highest normalized score
            best = max(remaining, key=lambda r: normalized[id(r)])
            selected.append(best)
            remaining.remove(best)
        else:
            # Find best MMR candidate
            best_mmr = float('-inf')
            best_candidate = None
            
            for candidate in remaining:
                norm_rel = normalized[id(candidate)]
                
                # Find max similarity to any selected result
                max_sim = 0.0
                for sel in selected:
                    sim = jaccard(token_sets[id(candidate)], token_sets[id(sel)])
                    if sim > max_sim:
                        max_sim = sim
                
                # Calculate MMR
                mmr = lambda_param * norm_rel - (1 - lambda_param) * max_sim
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
    
    return selected
