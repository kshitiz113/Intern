"""
Tool Resolver — Validates and orchestrates internal tool calls.
Enforces prerequisite graph (verify_identity before destructive actions).
"""

import json
import logging
from typing import List, Dict, Any

import config
from models import ToolCall

logger = logging.getLogger(__name__)

# ── Tool Schemas (loaded once) ────────────────────────────────────────

def _load_tool_schemas() -> List[Dict]:
    """Load tool schemas from internal_tools.json."""
    try:
        with open(config.TOOLS_SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Could not load tool schemas: {e}")
        return []

TOOL_SCHEMAS = _load_tool_schemas()
TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}
TOOL_REQUIRED_PARAMS = {}
for t in TOOL_SCHEMAS:
    TOOL_REQUIRED_PARAMS[t["name"]] = t.get("parameters", {}).get("required", [])

# ── Prerequisite Graph ────────────────────────────────────────────────
# Tools that require verify_identity before execution
REQUIRES_VERIFICATION = {"issue_refund", "lock_account", "modify_subscription"}


def get_tool_schemas_text() -> str:
    """Return tool schemas as a formatted string for LLM prompts."""
    lines = []
    for tool in TOOL_SCHEMAS:
        lines.append(f"- **{tool['name']}**: {tool['description']}")
        params = tool.get("parameters", {}).get("properties", {})
        required = tool.get("parameters", {}).get("required", [])
        for pname, pinfo in params.items():
            req_marker = " (REQUIRED)" if pname in required else ""
            lines.append(f"  - `{pname}`: {pinfo.get('description', '')}{req_marker}")
    return "\n".join(lines)


def validate_tool_calls(raw_actions: Any) -> List[Dict]:
    """
    Validate and fix a list of tool call dicts from LLM output.
    Enforces schema conformance and prerequisite graph.

    Args:
        raw_actions: List of dicts from LLM output, or string

    Returns:
        Validated list of tool call dicts
    """
    if not raw_actions:
        return []

    # Parse if string
    if isinstance(raw_actions, str):
        try:
            raw_actions = json.loads(raw_actions)
        except json.JSONDecodeError:
            return []

    if not isinstance(raw_actions, list):
        return []

    validated = []
    has_verify = False
    needs_verify = False

    for action in raw_actions:
        if not isinstance(action, dict):
            continue

        name = action.get("action", "")
        params = action.get("parameters", {})

        # Check if tool name is valid
        if name not in TOOL_NAMES:
            logger.warning(f"Unknown tool: {name}")
            continue

        # Check required parameters
        required = TOOL_REQUIRED_PARAMS.get(name, [])
        missing = [p for p in required if p not in params or not params[p]]
        if missing:
            logger.warning(f"Tool {name} missing required params: {missing}")
            # Still include but log the issue — the intent matters
            # Fill missing params with placeholder
            for p in missing:
                if p not in params:
                    params[p] = "unknown"

        if name == "verify_identity":
            has_verify = True

        if name in REQUIRES_VERIFICATION:
            needs_verify = True

        validated.append({"action": name, "parameters": params})

    # ── Enforce prerequisites ─────────────────────────────────────
    if needs_verify and not has_verify:
        # Auto-inject verify_identity at the beginning
        verify_call = {
            "action": "verify_identity",
            "parameters": {
                "method": "email_otp",
                "target": "user's registered email",
            },
        }
        validated.insert(0, verify_call)
        logger.info("Auto-injected verify_identity before destructive action")

    return validated


def format_actions_json(actions: List[Dict]) -> str:
    """Format tool calls as a JSON string for CSV output."""
    if not actions:
        return "[]"
    try:
        return json.dumps(actions, ensure_ascii=False)
    except Exception:
        return "[]"
