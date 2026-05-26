"""
Retriever — Hybrid BM25 + Semantic search with Reciprocal Rank Fusion.
Returns the most relevant corpus documents for a given query.
"""

import re
import logging
from typing import List, Optional

import numpy as np

import config
from models import RetrievedDoc
from corpus_indexer import CorpusIndexer

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (lexical) and semantic embedding search
    with Reciprocal Rank Fusion (RRF) to merge rankings.
    """

    def __init__(self, indexer: CorpusIndexer, llm_client=None):
        self._indexer = indexer
        self._llm = llm_client

    def retrieve(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        domain_filter: Optional[str] = None,
    ) -> List[RetrievedDoc]:
        """
        Retrieve top-k documents for a query using hybrid BM25 + semantic search.

        Args:
            query: Search query text
            top_k: Number of documents to return
            domain_filter: Optional product domain to filter by ("claude", "devplatform", "visa")
        """
        if not self._indexer.is_indexed or not query.strip():
            return []

        # ── BM25 search ───────────────────────────────────────────────
        bm25_ranked = self._bm25_search(query, top_n=20, domain_filter=domain_filter)

        # ── Semantic search ───────────────────────────────────────────
        semantic_ranked = self._semantic_search(query, top_n=20, domain_filter=domain_filter)

        # ── Reciprocal Rank Fusion ────────────────────────────────────
        if semantic_ranked:
            fused = self._rrf_merge(bm25_ranked, semantic_ranked, k=config.RRF_K)
        else:
            # Fallback to BM25 only if embeddings unavailable
            fused = bm25_ranked

        # ── Build result list ─────────────────────────────────────────
        results = []
        seen_paths = set()
        for idx, score in fused[:top_k]:
            path = self._indexer.doc_paths[idx]
            if path in seen_paths:
                continue
            seen_paths.add(path)

            # Truncate document content for context window management
            content = self._indexer.documents[idx]
            if len(content) > 3000:
                content = content[:3000] + "\n... [truncated]"

            results.append(RetrievedDoc(
                content=content,
                path=path,
                score=score,
                domain=self._indexer.doc_domains[idx],
            ))

        return results

    def _bm25_search(
        self, query: str, top_n: int = 20, domain_filter: Optional[str] = None
    ) -> List[tuple]:
        """BM25 lexical search. Returns list of (doc_index, score)."""
        tokens = re.findall(r'[a-z0-9]+', query.lower())
        if not tokens or self._indexer.bm25 is None:
            return []

        scores = self._indexer.bm25.get_scores(tokens)

        # Apply domain filter
        if domain_filter:
            for i, domain in enumerate(self._indexer.doc_domains):
                if domain != domain_filter:
                    scores[i] = 0.0

        # Get top-N indices
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def _semantic_search(
        self, query: str, top_n: int = 20, domain_filter: Optional[str] = None
    ) -> List[tuple]:
        """Semantic embedding search. Returns list of (doc_index, score)."""
        if self._indexer.embeddings is None or self._llm is None:
            return []

        try:
            query_embedding = np.array(self._llm.embed_single(query), dtype=np.float32)
            similarities = _cosine_similarity(query_embedding, self._indexer.embeddings)

            # Apply domain filter
            if domain_filter:
                for i, domain in enumerate(self._indexer.doc_domains):
                    if domain != domain_filter:
                        similarities[i] = -1.0

            top_indices = np.argsort(similarities)[::-1][:top_n]
            return [
                (int(idx), float(similarities[idx]))
                for idx in top_indices
                if similarities[idx] > 0
            ]
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    @staticmethod
    def _rrf_merge(
        ranking_a: List[tuple],
        ranking_b: List[tuple],
        k: int = 60,
    ) -> List[tuple]:
        """
        Reciprocal Rank Fusion: merges two rankings into one.
        score(doc) = sum(1 / (k + rank_i(doc))) for each ranking i
        """
        scores = {}

        for rank, (idx, _) in enumerate(ranking_a):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

        for rank, (idx, _) in enumerate(ranking_b):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

        # Sort by fused score descending
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused
