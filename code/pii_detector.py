"""
PII Detector — Identifies personally identifiable information in ticket text.
Uses regex patterns for structured PII (credit cards, SSNs, etc.)
The LLM handles contextual PII detection as part of its reasoning call.
"""

import re
from typing import List, Tuple

from models import PIIResult


# ── Regex Patterns ────────────────────────────────────────────────────

# Credit card: 13-19 digits, optionally separated by spaces or dashes
_CC_PATTERN = re.compile(
    r'\b(?:\d[ -]*?){13,19}\b'
)
# Strict CC: common prefixes (Visa 4xxx, MC 5xxx, Amex 3xxx)
_CC_STRICT = re.compile(
    r'\b(?:4[0-9]{3}|5[1-5][0-9]{2}|3[47][0-9]{2}|6(?:011|5[0-9]{2}))'
    r'[ -]?[0-9]{4}[ -]?[0-9]{4}[ -]?[0-9]{1,4}\b'
)

# SSN: XXX-XX-XXXX
_SSN_PATTERN = re.compile(
    r'\b(?!000|666|9\d\d)\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b'
)

# Email
_EMAIL_PATTERN = re.compile(
    r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b'
)

# Phone: international formats
_PHONE_PATTERN = re.compile(
    r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'
)

# Physical address heuristic: number + street name + common suffix
_ADDRESS_PATTERN = re.compile(
    r'\b\d{1,5}\s+[A-Z][a-zA-Z]+\s+'
    r'(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl)'
    r'[\s,]',
    re.IGNORECASE,
)

# Date of birth patterns
_DOB_PATTERN = re.compile(
    r'\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b'
)

# IP Address
_IP_PATTERN = re.compile(
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
)


def _luhn_check(number_str: str) -> bool:
    """Validate a number string using the Luhn algorithm."""
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def detect_pii(text: str) -> PIIResult:
    """
    Scan text for PII using regex patterns.
    Returns a PIIResult with detection flag and found PII types.
    """
    if not text:
        return PIIResult()

    found_types: List[str] = []
    found_details: List[str] = []

    # Credit cards (with Luhn validation)
    for match in _CC_STRICT.finditer(text):
        raw = match.group()
        digits_only = re.sub(r'[^\d]', '', raw)
        if _luhn_check(digits_only):
            found_types.append("credit_card")
            found_details.append(f"Card number detected (ending ...{digits_only[-4:]})")

    # SSN
    for match in _SSN_PATTERN.finditer(text):
        raw = match.group()
        digits_only = re.sub(r'[^\d]', '', raw)
        # Avoid matching phone numbers or other digit sequences
        if len(digits_only) == 9 and '-' in raw:
            found_types.append("ssn")
            found_details.append("SSN detected")

    # Email addresses
    for match in _EMAIL_PATTERN.finditer(text):
        found_types.append("email")
        found_details.append(f"Email: {match.group()}")

    # Phone numbers — only flag if it looks like a personal phone
    phone_matches = _PHONE_PATTERN.findall(text)
    # Filter out obvious non-phone numbers (too short, or support hotlines)
    for phone in phone_matches:
        digits = re.sub(r'[^\d]', '', phone)
        if 10 <= len(digits) <= 15:
            found_types.append("phone_number")
            found_details.append(f"Phone number detected")
            break  # Only flag once

    # Physical addresses
    if _ADDRESS_PATTERN.search(text):
        found_types.append("address")
        found_details.append("Physical address detected")

    # Date of birth
    if _DOB_PATTERN.search(text):
        found_types.append("date_of_birth")
        found_details.append("Date of birth detected")

    # Deduplicate types
    unique_types = list(dict.fromkeys(found_types))

    return PIIResult(
        pii_detected=len(unique_types) > 0,
        pii_types=unique_types,
        details=found_details,
    )
