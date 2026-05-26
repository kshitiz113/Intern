"""
LLM Client — Wrapper around Google Gemini API.
Handles generation (structured JSON output) and embeddings.
"""

import re
import json
import time
import logging
from typing import List, Optional, Any

from google import genai
from google.genai import types as genai_types

import config

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified client for Gemini generation and embedding."""

    def __init__(self):
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self._call_count = 0
        self._embed_cache = {}

    # ── Text Generation ───────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = True,
    ) -> str:
        """
        Generate text using Gemini.
        Returns raw text (parsed as JSON by the caller).
        """
        gen_config = genai_types.GenerateContentConfig(
            temperature=config.LLM_TEMPERATURE,
            seed=config.LLM_SEED,
            max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        if system_prompt:
            gen_config.system_instruction = system_prompt

        for attempt in range(config.MAX_RETRIES):
            try:
                self._call_count += 1
                response = self._client.models.generate_content(
                    model=config.LLM_MODEL,
                    contents=prompt,
                    config=gen_config,
                )
                # Handle safety blocks / empty responses
                if response.text:
                    return response.text.strip()
                # If blocked by safety filters, return a safe default
                logger.warning("Empty LLM response (possible safety block)")
                return json.dumps(self._safe_default_response())

            except Exception as e:
                is_rate_limit = "429" in str(e) or "resource_exhausted" in str(e).lower() or "quota" in str(e).lower()
                delay = 15.0 if is_rate_limit else config.RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Retrying in {delay}s")
                time.sleep(delay)

        logger.error("All LLM retries exhausted, returning safe default")
        return json.dumps(self._safe_default_response())

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        """Generate plain text (non-JSON) response."""
        return self.generate(prompt, system_prompt, json_mode=False)

    # ── Embeddings ────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using Gemini.
        Caches results to avoid redundant API calls.
        Returns list of embedding vectors.
        """
        if not texts:
            return []

        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for idx, text in enumerate(texts):
            if text in self._embed_cache:
                results[idx] = self._embed_cache[text]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            computed_embeddings = []
            # Process in batches
            for i in range(0, len(uncached_texts), config.EMBEDDING_BATCH_SIZE):
                if i > 0:
                    time.sleep(4.5)  # Sleep between batches to stay under 15 RPM
                batch_texts = uncached_texts[i : i + config.EMBEDDING_BATCH_SIZE]
                batch_contents = [
                    genai_types.Content(parts=[genai_types.Part.from_text(text=txt)])
                    for txt in batch_texts
                ]
                batch_embeddings = []
                for attempt in range(config.MAX_RETRIES):
                    try:
                        self._call_count += 1
                        result = self._client.models.embed_content(
                            model=config.EMBEDDING_MODEL,
                            contents=batch_contents,
                        )
                        batch_embeddings = [e.values for e in result.embeddings]
                        break
                    except Exception as e:
                        is_rate_limit = "429" in str(e) or "resource_exhausted" in str(e).lower() or "quota" in str(e).lower()
                        delay = 15.0 if is_rate_limit else config.RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            f"Embedding call failed (attempt {attempt+1}): {e}. "
                            f"Retrying in {delay}s"
                        )
                        time.sleep(delay)
                else:
                    # All retries failed — use zero vectors as fallback
                    logger.error(f"Embedding failed for batch starting at {i}, using zeros")
                    batch_embeddings = [[0.0] * 3072] * len(batch_texts)
                
                computed_embeddings.extend(batch_embeddings)

            # Store in cache and populate results
            for idx, text, emb in zip(uncached_indices, uncached_texts, computed_embeddings):
                self._embed_cache[text] = emb
                results[idx] = emb

        return results

    def embed_single(self, text: str) -> List[float]:
        """Embed a single text. Convenience wrapper with whitespace standardization to maximize cache hits."""
        clean_text = re.sub(r'\s+', ' ', text).strip()
        results = self.embed([clean_text])
        return results[0] if results else [0.0] * 3072

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _safe_default_response() -> dict:
        """Fallback response when LLM completely fails."""
        return {
            "status": "escalated",
            "product_area": "general",
            "request_type": "product_issue",
            "response": "I apologize, but I'm unable to fully process your request at this time. I've escalated your ticket to a human support agent who will be able to assist you directly.",
            "justification": "Agent was unable to generate a confident response. Escalating to human support for safety.",
            "confidence_score": 0.2,
            "risk_level": "medium",
            "pii_detected": "false",
            "language": "en",
            "actions_taken": [{"action": "escalate_to_human", "parameters": {"priority": "normal", "department": "general", "summary": "Agent processing failure — requires human review"}}],
            "source_documents": [],
        }

    @property
    def call_count(self) -> int:
        return self._call_count
