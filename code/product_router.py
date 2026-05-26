"""
Product Router — Determines the actual product domain for a ticket.
Content-based routing that doesn't blindly trust the 'company' field.
"""

import re
import logging
from typing import Optional

from models import Conversation

logger = logging.getLogger(__name__)

# ── Domain Keyword Signals ────────────────────────────────────────────

_DEVPLATFORM_KEYWORDS = {
    'devplatform', 'hackerrank', 'codepair', 'codescreen', 'assessment',
    'test', 'coding test', 'candidate', 'interviewer', 'proctoring',
    'proctor', 'code editor', 'mock interview', 'skillup', 'certif',
    'resume builder', 'submission', 'challenge', 'badge',
    'hiring', 'recruiter', 'apply tab', 'interview',
}

_CLAUDE_KEYWORDS = {
    'claude', 'anthropic', 'ai assistant', 'conversation history',
    'chat', 'project', 'artifact', 'claude pro', 'claude team',
    'claude enterprise', 'bedrock', 'api key', 'model', 'token',
    'prompt', 'claude code', 'claude desktop', 'connector', 'mcp',
    'lti', 'web crawl',
}

_VISA_KEYWORDS = {
    'visa', 'card', 'credit card', 'debit card', 'transaction',
    'payment', 'merchant', 'chargeback', 'refund', 'atm',
    'pin', 'traveller', 'cheque', 'fraud', 'unauthorized',
    'dispute', 'blocked card', 'zero liability', 'contactless',
    'travel', 'cash advance', 'cardholder',
}


def _count_keyword_matches(text: str, keywords: set) -> int:
    """Count how many domain keywords appear in the text."""
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        if kw in text_lower:
            count += 1
    return count


def route_product(
    conversation: Conversation,
    company_field: str,
    subject: str = "",
) -> Optional[str]:
    """
    Determine the actual product domain for a ticket.
    Uses content analysis over the company field when there's a mismatch.

    Returns: "claude", "devplatform", "visa", or None (ambiguous/unknown)
    """
    full_text = f"{subject} {conversation.full_text}".strip()
    if not full_text:
        # No content to analyze
        return _normalize_company(company_field)

    # Count keyword matches for each domain
    dp_score = _count_keyword_matches(full_text, _DEVPLATFORM_KEYWORDS)
    cl_score = _count_keyword_matches(full_text, _CLAUDE_KEYWORDS)
    vi_score = _count_keyword_matches(full_text, _VISA_KEYWORDS)

    # If clear winner from keywords
    scores = {"devplatform": dp_score, "claude": cl_score, "visa": vi_score}
    max_score = max(scores.values())

    if max_score == 0:
        # No keywords matched — trust company field
        return _normalize_company(company_field)

    winners = [k for k, v in scores.items() if v == max_score]

    if len(winners) == 1:
        # Clear winner from content analysis
        content_domain = winners[0]
        company_domain = _normalize_company(company_field)

        if company_domain and company_domain != content_domain:
            logger.info(
                f"Content-based routing overrides company field: "
                f"'{company_field}' → '{content_domain}'"
            )
        return content_domain

    # Tie or ambiguous — use company field as tiebreaker
    company_domain = _normalize_company(company_field)
    if company_domain and company_domain in winners:
        return company_domain

    # Still ambiguous — return first winner
    return winners[0] if winners else None


def _normalize_company(company: str) -> Optional[str]:
    """Normalize the company field to a standard domain name."""
    if not company:
        return None
    c = company.strip().lower()
    if c in ("none", "null", ""):
        return None
    if "devplatform" in c or "hackerrank" in c:
        return "devplatform"
    if "claude" in c or "anthropic" in c:
        return "claude"
    if "visa" in c:
        return "visa"
    return None
