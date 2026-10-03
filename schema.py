
from typing import Literal

from pydantic import BaseModel, ValidationError

Severity = Literal["critical", "high", "medium", "low"]
Category = Literal["security", "infrastructure", "application", "data", "network"]
ReportedBy = Literal[
    "on-call engineer", "SRE", "customer support agent",
    "security analyst", "devops engineer", "qa tester", "unspecified",
]
TimeDetected = Literal[
    "just now", "within the last hour", "this morning",
    "this afternoon", "overnight", "yesterday", "unspecified",
]
ActionTaken = Literal[
    "none", "restarted the service", "rolled back the deployment",
    "escalated to on-call", "applied a temporary workaround",
    "disabled the affected feature", "notified the vendor",
]

AFFECTED_SYSTEMS = [
    "payments-api", "auth-service", "customer-db", "checkout-frontend",
    "email-gateway", "vpn-gateway", "backup-storage", "ci-cd-pipeline",
    "internal-wiki", "load-balancer", "search-index", "notification-service",
]


# Shared field-list text for the system prompt, used by every file that
# needs to tell the model what to extract (data generation already builds
# its own instructions separately, but eval_harness.py, predictor.py, the
# training notebook, and app/gradio_app.py all previously duplicated this
# exact text -- which is exactly how the "other" category removal needed
# patching in 4 separate places. Single source of truth now.
SYSTEM_PROMPT_FIELDS = """- severity: one of "critical", "high", "medium", "low"
- category: one of "security", "infrastructure", "application", "data", "network"
- affected_system: the name of the affected system as mentioned in the report
- reported_by: one of "on-call engineer", "SRE", "customer support agent", "security analyst", "devops engineer", "qa tester", "unspecified"
- time_detected: one of "just now", "within the last hour", "this morning", "this afternoon", "overnight", "yesterday", "unspecified"
- action_taken: one of "none", "restarted the service", "rolled back the deployment", "escalated to on-call", "applied a temporary workaround", "disabled the affected feature", "notified the vendor"
- requires_escalation: true or false
- customer_impact: true or false"""


class IncidentReport(BaseModel):
    severity: Severity
    category: Category
    affected_system: str  # validated against AFFECTED_SYSTEMS by the caller (kept as str so near-miss spellings can still be scored as partial credit if desired)
    reported_by: ReportedBy
    time_detected: TimeDetected
    action_taken: ActionTaken
    requires_escalation: bool
    customer_impact: bool


def parse_model_output(raw_text: str) -> IncidentReport | None:
    """
    Attempts to parse a model's raw text output into a validated
    IncidentReport. Strips common formatting issues (markdown code
    fences, leading/trailing prose) before giving up. Returns None
    rather than raising, so callers (eval harness, inference wrapper)
    can treat "failed to parse" as a first-class, countable outcome
    instead of a crash.
    """
    import json
    import re

    text = raw_text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        # Fall back to the first {...} block found anywhere in the text,
        # in case the model added commentary before/after the JSON.
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)

    try:
        data = json.loads(text)
        return IncidentReport(**data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None