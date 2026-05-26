"""
Central configuration for the Support Triage Agent.
All paths, constants, and tunable parameters in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
TICKETS_DIR = PROJECT_ROOT / "support_tickets"
INPUT_CSV = TICKETS_DIR / "support_tickets.csv"
OUTPUT_CSV = TICKETS_DIR / "output.csv"
TOOLS_SCHEMA_PATH = DATA_DIR / "api_specs" / "internal_tools.json"
EMBEDDINGS_CACHE = DATA_DIR / ".embeddings_cache.pkl"

# ── LLM Settings ──────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
LLM_TEMPERATURE = 0.0
LLM_SEED = 42
LLM_MAX_OUTPUT_TOKENS = 8192

# ── Retrieval Settings ────────────────────────────────────────────────
RETRIEVAL_TOP_K = 5
BM25_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6
RRF_K = 60  # Reciprocal Rank Fusion constant
EMBEDDING_BATCH_SIZE = 100  # texts per embedding API call

# ── Safety Settings ───────────────────────────────────────────────────
ANOMALY_THRESHOLD = 0.35  # cosine distance for embedding anomaly detection

# ── Processing Settings ───────────────────────────────────────────────
PER_TICKET_TIMEOUT = 45  # seconds
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds, exponential backoff base

# ── Output Column Order ──────────────────────────────────────────────
OUTPUT_COLUMNS = [
    "issue", "subject", "company", "response", "product_area",
    "status", "request_type", "justification", "confidence_score",
    "source_documents", "risk_level", "pii_detected", "language",
    "actions_taken",
]

# ── Valid Enum Values ─────────────────────────────────────────────────
VALID_STATUS = {"replied", "escalated"}
VALID_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}
VALID_RISK_LEVEL = {"low", "medium", "high", "critical"}
VALID_PII = {"true", "false"}
