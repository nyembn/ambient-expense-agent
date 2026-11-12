# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ambient agent that processes expense reports via an ADK 2.0 graph workflow.

This agent receives incoming expense report events and routes them:
- Expenses under $200: Auto-approved immediately in code, no LLM involved.
- Expenses of $200 or more: Reviewed by an LLM agent for risk factors with an alert,
  then paused for human-in-the-loop (HITL) approval via RequestInput.
"""

import base64
import json
import re
from typing import Any

from google.adk import Agent, Context, Event, Workflow
from google.adk.apps import App
from google.adk.events import RequestInput
from google.genai import types
from pydantic import BaseModel, Field

from .config import config


# ---------------------------------------------------------------------------
# Pydantic schemas for structured data flow between nodes
# ---------------------------------------------------------------------------


class ExpenseData(BaseModel):
    """Expense report data extracted from incoming JSON event."""

    amount: float = Field(description="Expense amount in USD")
    submitter: str = Field(description="Email or username of the person who submitted")
    category: str = Field(description="Expense category, e.g. travel, meals, equipment")
    description: str = Field(description="What the expense is for")
    date: str = Field(description="Date of the expense (YYYY-MM-DD)")


# ---------------------------------------------------------------------------
# Function nodes
# ---------------------------------------------------------------------------


def parse_expense_event(node_input: Any) -> Event:
    """Parse incoming trigger event and extract expense data.

    The expense payload sits under a 'data' key that might be base64-encoded
    (real Pub/Sub) or plain JSON / dict (local testing).
    """
    if hasattr(node_input, "parts") and node_input.parts:
        text = node_input.parts[0].text or ""
    elif isinstance(node_input, str):
        text = node_input
    elif isinstance(node_input, dict):
        text = None
        event = node_input
    else:
        text = str(node_input)

    if text is not None:
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            return Event(output={"error": f"Invalid JSON: {text[:200]}"})

    if isinstance(event, dict) and "message" in event and isinstance(event["message"], dict):
        event = event["message"]

    data = event.get("data", event)

    if isinstance(data, str):
        try:
            decoded = base64.b64decode(data).decode("utf-8")
            data = json.loads(decoded)
        except Exception:
            try:
                data = json.loads(data)
            except Exception:
                return Event(output={"error": f"Failed to decode data: {data[:200]}"})

    expense = {
        "amount": float(data.get("amount", 0.0)),
        "submitter": str(data.get("submitter", "unknown")),
        "category": str(data.get("category", "other")),
        "description": str(data.get("description", "")),
        "date": str(data.get("date", "")),
    }
    return Event(output=expense)


def route_by_amount(node_input: dict[str, Any], ctx: Context) -> Event:
    """Route expenses based on the threshold in Python code.

    - Under $100 -> AUTO_APPROVE (instant, no LLM)
    - $100 or more -> NEEDS_REVIEW (routes to LLM review agent)
    """
    ctx.state["expense_data"] = node_input
    amount = node_input.get("amount", 0.0)
    if amount >= config.review_threshold:
        return Event(route="NEEDS_REVIEW", output=node_input)
    return Event(route="AUTO_APPROVE", output=node_input)


def auto_approve(node_input: dict[str, Any]) -> Event:
    """Auto-approve a low-value expense and log the decision."""
    amount = float(node_input.get("amount", 0.0))
    submitter = str(node_input.get("submitter", "unknown"))
    category = str(node_input.get("category", "other"))
    date = str(node_input.get("date", ""))

    log_entry = {
        "severity": "INFO",
        "message": f"Expense auto-approved: ${amount:.2f} from {submitter}",
        "decision": "approved",
        "amount": amount,
        "submitter": submitter,
        "category": category,
    }
    print(json.dumps(log_entry), flush=True)

    summary_text = (
        f"Expense auto-approved: ${amount:.2f} from {submitter} "
        f"({category}) on {date}."
    )
    return Event(
        output={"status": "approved", **node_input},
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=summary_text)],
        ),
    )


# ---------------------------------------------------------------------------
# Security checkpoint: PII scrubbing and prompt injection defense
# ---------------------------------------------------------------------------

SSN_PATTERN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4,6}\b")
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b"
)

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|any\s+|prior\s+|previous\s+)?instructions", re.IGNORECASE),
    re.compile(r"auto-?approve", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+|the\s+)?rules?", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"override", re.IGNORECASE),
    re.compile(r"disregard", re.IGNORECASE),
    re.compile(r"approve\s+(this\s+|immediately|without\s+review)", re.IGNORECASE),
    re.compile(r"do\s+not\s+(review|flag|check)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
]


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Scrub SSNs and credit card numbers from text and return redacted categories."""
    redacted_categories: list[str] = []
    if SSN_PATTERN.search(text):
        redacted_categories.append("ssn")
        text = SSN_PATTERN.sub("[REDACTED_SSN]", text)
    if CREDIT_CARD_PATTERN.search(text):
        redacted_categories.append("credit_card")
        text = CREDIT_CARD_PATTERN.sub("[REDACTED_CREDIT_CARD]", text)
    return text, redacted_categories


def is_prompt_injection(text: str) -> bool:
    """Detect if text contains prompt injection or rule-bypass instructions."""
    return any(pattern.search(text) for pattern in PROMPT_INJECTION_PATTERNS)


def security_checkpoint(node_input: dict[str, Any], ctx: Context) -> Event:
    """Security checkpoint to scrub PII and defend against prompt injection.

    - Scrubs personal data (SSNs, credit card numbers) from description.
    - Records redacted categories in state.
    - Inspects description for prompt injection.
    - If prompt injection detected: logs security event and routes directly to
      human approval (bypassing the LLM).
    - If clean: routes to LLM review agent.
    """
    expense = dict(node_input)
    original_description = expense.get("description", "")

    # 1. Scrub PII
    cleaned_description, redacted_categories = scrub_pii(original_description)
    expense["description"] = cleaned_description
    if redacted_categories:
        expense["redacted_categories"] = redacted_categories

    # Persist scrubbed data and redactions in context
    ctx.state["expense_data"] = expense
    ctx.state["redacted_categories"] = redacted_categories

    # 2. Defend against prompt injection
    if is_prompt_injection(original_description):
        log_entry = {
            "severity": "CRITICAL",
            "alert_type": "security_violation",
            "event": "prompt_injection_detected",
            "submitter": expense.get("submitter", "unknown"),
            "amount": expense.get("amount", 0.0),
            "message": (
                "Prompt injection attempt detected in expense description. "
                "Bypassing LLM review and routing directly to manager."
            ),
        }
        print(json.dumps(log_entry), flush=True)

        ctx.state["security_flag"] = (
            "Potential prompt injection detected in description. LLM review bypassed."
        )
        return Event(route="SECURITY_FLAGGED", output=expense)

    return Event(route="CLEAN", output=expense)


# ---------------------------------------------------------------------------
# LLM review agent (invoked only for expenses >= $100)
# ---------------------------------------------------------------------------


def emit_expense_alert(
    submitter: str,
    amount: float,
    category: str,
    risk_summary: str,
) -> dict[str, Any]:
    """Emit a structured log alerting finance to review a high-value expense.

    Args:
        submitter: Who submitted the expense.
        amount: The expense amount in USD.
        category: The expense category.
        risk_summary: Why this expense needs human review.

    Returns:
        Confirmation that the alert was emitted.
    """
    log_entry = {
        "severity": "WARNING",
        "message": (
            f"Expense review alert: ${amount:.2f} from {submitter} — {risk_summary}"
        ),
        "alert_type": "expense_review",
        "submitter": submitter,
        "amount": amount,
        "category": category,
        "risk_summary": risk_summary,
    }
    print(json.dumps(log_entry), flush=True)
    return {"status": "alert_emitted", "submitter": submitter, "amount": amount}


review_agent = Agent(
    name="review_agent",
    model=config.model,
    mode="single_turn",
    instruction="""You are an expense review agent. You receive expense reports
of $100 or more that need review before approval.

Analyze the expense and:
1. Check for risk factors: unusual category for the amount, vague description,
   suspiciously round numbers, very high value (>$1000), or potential policy violations.
2. Call the `emit_expense_alert` tool with the submitter, amount, category,
   and a brief risk summary explaining why this expense needs human review.
3. Return a structured review.

Your review MUST include:
- **Amount**: The expense amount
- **Submitter**: Who submitted it
- **Category**: The expense category
- **Risk level**: low, medium, or high
- **Risk factors**: What flags you found (if any)
- **Recommendation**: approve, request-more-info, or escalate""",
    input_schema=ExpenseData,
    tools=[emit_expense_alert],
)


# ---------------------------------------------------------------------------
# HITL: pause the workflow for human approval
# ---------------------------------------------------------------------------


def request_approval(node_input: Any, ctx: Context):
    """Pause the workflow and wait for a human to approve or reject.

    Yields a RequestInput that pauses the workflow. The human's response
    flows into process_decision on resume.
    """
    expense = ctx.state.get("expense_data", {})
    security_flag = ctx.state.get("security_flag")
    redacted_categories = ctx.state.get("redacted_categories", [])

    payload = dict(expense)
    if security_flag:
        payload["security_alert"] = security_flag
    if redacted_categories:
        payload["redacted_categories"] = redacted_categories

    message = (
        f"SECURITY ALERT: {security_flag} Manager approval required."
        if security_flag
        else "Expense requires manager approval. Approve or reject."
    )
    yield RequestInput(
        message=message,
        payload=payload,
    )


def process_decision(node_input: Any, ctx: Context) -> Event:
    """Process the human's approval decision and log/record the outcome."""
    decision = "unknown"
    if isinstance(node_input, dict):
        decision = node_input.get("decision", "unknown")
    elif isinstance(node_input, str):
        decision = "approve" if "approve" in node_input.lower() else "reject"

    approved = decision == "approve"
    expense = ctx.state.get("expense_data", {})
    status = "approved" if approved else "rejected"

    log_entry = {
        "severity": "INFO" if approved else "WARNING",
        "message": f"Expense {status} by manager",
        "decision": status,
    }
    print(json.dumps(log_entry), flush=True)

    submitter = expense.get("submitter", "unknown")
    amount = expense.get("amount", 0.0)
    category = expense.get("category", "")
    description = expense.get("description", "")
    date = expense.get("date", "")

    parts = [f"${amount:.2f} expense from {submitter} has been {status}."]
    if description:
        parts.append(f'"{description}" ({category}) on {date}.')
    if approved:
        parts.append("The expense has been logged and will be processed for reimbursement.")
    else:
        parts.append("The submitter will be notified and may resubmit with additional documentation.")

    msg = " ".join(parts)
    return Event(
        output={"status": status, "message": msg},
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=msg)],
        ),
    )


# ---------------------------------------------------------------------------
# Graph-based workflow — the root agent
# ---------------------------------------------------------------------------

root_agent = Workflow(
    name="expense_processor",
    edges=[
        ("START", parse_expense_event, route_by_amount),
        (
            route_by_amount,
            {
                "AUTO_APPROVE": auto_approve,
                "NEEDS_REVIEW": security_checkpoint,
            },
        ),
        (
            security_checkpoint,
            {
                "CLEAN": review_agent,
                "SECURITY_FLAGGED": request_approval,
            },
        ),
        (review_agent, request_approval, process_decision),
    ],
)

app = App(
    name="app",
    root_agent=root_agent,
)
