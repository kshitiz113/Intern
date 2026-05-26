"""
Data models for the Support Triage Agent.
All structured types used across the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    """A single message in a conversation."""
    role: str
    content: str


@dataclass
class Conversation:
    """Parsed multi-turn conversation from a ticket's issue field."""
    messages: List[Message] = field(default_factory=list)
    raw_text: str = ""

    @property
    def latest_user_message(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return self.raw_text

    @property
    def full_text(self) -> str:
        if not self.messages:
            return self.raw_text
        return "\n".join(f"[{m.role}]: {m.content}" for m in self.messages)

    @property
    def is_empty(self) -> bool:
        return len(self.messages) == 0 and not self.raw_text.strip()

    @property
    def is_multi_turn(self) -> bool:
        return len(self.messages) > 1


@dataclass
class Ticket:
    """A single support ticket to process."""
    issue_raw: str
    subject: str
    company: str
    conversation: Optional[Conversation] = None
    row_index: int = 0  # 1-based row in CSV for logging


@dataclass
class SafetyResult:
    """Output of the adversarial safety analysis."""
    is_adversarial: bool = False
    attack_type: str = "none"
    confidence: float = 0.0
    reasoning: str = ""
    has_legitimate_request: bool = False
    legitimate_request: str = ""


@dataclass
class PIIResult:
    """Output of PII detection."""
    pii_detected: bool = False
    pii_types: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)


@dataclass
class RetrievedDoc:
    """A single document retrieved from the corpus."""
    content: str
    path: str          # relative to project root (e.g., "data/claude/...")
    score: float = 0.0
    domain: str = ""   # "claude" | "devplatform" | "visa"


@dataclass
class ToolCall:
    """A single internal tool call."""
    action: str
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"action": self.action, "parameters": self.parameters}


@dataclass
class TicketResult:
    """Complete output for one ticket — maps 1:1 to an output CSV row."""
    # Input fields (preserved from original)
    issue: str = ""
    subject: str = ""
    company: str = ""
    # Generated fields
    response: str = ""
    product_area: str = ""
    status: str = "replied"
    request_type: str = "product_issue"
    justification: str = ""
    confidence_score: float = 0.5
    source_documents: str = ""
    risk_level: str = "medium"
    pii_detected: str = "false"
    language: str = "en"
    actions_taken: str = "[]"
