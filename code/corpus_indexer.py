"""
Corpus Indexer — Builds BM25 and semantic embedding indices for all corpus docs.
Caches embeddings to disk to avoid recomputation.
"""

import os
import re
import pickle
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from rank_bm25 import BM25Okapi

import config

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]+', text)
    return tokens


def _detect_domain(path: str) -> str:
    """Determine product domain from file path."""
    path_lower = path.lower().replace("\\", "/")
    if "/claude/" in path_lower or "/claude-" in path_lower:
        return "claude"
    elif "/devplatform/" in path_lower or "/hackerrank" in path_lower:
        return "devplatform"
    elif "/visa/" in path_lower:
        return "visa"
    return "unknown"


def _compute_cache_key(file_paths: List[str]) -> str:
    """Compute a hash of file paths + sizes for cache invalidation."""
    parts = []
    for fp in sorted(file_paths):
        try:
            size = os.path.getsize(fp)
            parts.append(f"{fp}:{size}")
        except OSError:
            parts.append(fp)
    return hashlib.md5("|".join(parts).encode()).hexdigest()


class CorpusIndexer:
    """
    Indexes all .md files under data/ for retrieval.
    Builds both BM25 (lexical) and embedding (semantic) indices.
    """

    def __init__(self, llm_client=None):
        self._llm = llm_client
        self.documents: List[str] = []       # full text of each doc
        self.doc_paths: List[str] = []        # relative paths
        self.doc_domains: List[str] = []      # product domain per doc
        self.doc_titles: List[str] = []       # first heading or filename
        self.bm25: BM25Okapi = None
        self.embeddings: np.ndarray = None    # (N, dim) matrix
        self._indexed = False

    def build_index(self) -> None:
        """Load all corpus files and build BM25 + embedding indices."""
        logger.info("Building corpus index...")

        # ── Load all documents ────────────────────────────────────────
        self._load_documents()
        if not self.documents:
            logger.error("No corpus documents found!")
            return

        logger.info(f"Loaded {len(self.documents)} corpus documents")

        # ── Build BM25 index ──────────────────────────────────────────
        tokenized = [_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built")

        # ── Build / load embedding index ──────────────────────────────
        self._build_embedding_index()
        self._indexed = True
        logger.info("Corpus indexing complete")

    def _load_documents(self) -> None:
        """Read all .md files from data/ directory."""
        data_dir = config.DATA_DIR

        for root, _dirs, files in os.walk(data_dir):
            for fname in sorted(files):
                if not fname.endswith(".md"):
                    continue

                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, config.PROJECT_ROOT).replace("\\", "/")

                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read().strip()
                except Exception as e:
                    logger.warning(f"Could not read {fpath}: {e}")
                    continue

                if not content:
                    continue

                # Extract title from first heading or filename
                title = fname.replace(".md", "").replace("-", " ").title()
                for line in content.split("\n"):
                    if line.startswith("# "):
                        title = line.lstrip("# ").strip()
                        break

                domain = _detect_domain(rel_path)

                # Also check content for domain cues if path says "visa" but content is about devplatform
                content_lower = content.lower()
                if domain == "visa" and ("devplatform" in content_lower or "hackerrank" in content_lower):
                    if "visa" not in content_lower and "card" not in content_lower:
                        domain = "devplatform"  # Miscategorized document

                self.documents.append(content)
                self.doc_paths.append(rel_path)
                self.doc_domains.append(domain)
                self.doc_titles.append(title)

    def _build_embedding_index(self) -> None:
        """Build or load cached embedding index."""
        cache_path = config.EMBEDDINGS_CACHE
        cache_key = _compute_cache_key(
            [os.path.join(config.PROJECT_ROOT, p) for p in self.doc_paths]
        )

        # Try loading from cache
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("key") == cache_key and len(cached.get("embeddings", [])) == len(self.documents):
                    self.embeddings = np.array(cached["embeddings"])
                    logger.info(f"Loaded embeddings from cache ({self.embeddings.shape})")
                    return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")

        # Compute embeddings
        if self._llm is None:
            logger.warning("No LLM client — skipping embeddings, using BM25 only")
            self.embeddings = None
            return

        logger.info(f"Computing embeddings for {len(self.documents)} documents...")

        # Prepare texts for embedding (title + first 500 chars of content)
        embed_texts = []
        for i, doc in enumerate(self.documents):
            title = self.doc_titles[i]
            # Use title + truncated content for embedding
            text = f"{title}\n{doc[:2000]}"
            embed_texts.append(text)

        try:
            raw_embeddings = self._llm.embed(embed_texts)
            self.embeddings = np.array(raw_embeddings, dtype=np.float32)

            # Save to cache
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "key": cache_key,
                        "embeddings": raw_embeddings,
                    }, f)
                logger.info(f"Embeddings cached to {cache_path}")
            except Exception as e:
                logger.warning(f"Could not cache embeddings: {e}")

        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            self.embeddings = None

    @property
    def is_indexed(self) -> bool:
        return self._indexed

    @property
    def size(self) -> int:
        return len(self.documents)
