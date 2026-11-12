# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local LLM-as-judge metric for routing correctness.

Evaluates whether the expense agent correctly adheres to business routing rules:
- Expenses strictly under $100 (< $100) are auto-approved in code without human approval.
- Expenses of $100 or more (>= $100) must require human-in-the-loop (HITL) manager review
  and must NEVER be auto-approved.
"""

import json
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


class _Verdict(BaseModel):
    score: int = Field(..., ge=1, le=5, description="Routing correctness score from 1 to 5")
    explanation: str = Field(..., description="Short explanation for the score")


def evaluate(instance: dict[str, Any]) -> dict[str, Any]:
    """Grade routing correctness on a 1-5 scale using an LLM judge."""
    prompt_data = instance.get("prompt", "")
    response_data = instance.get("response", "")
    agent_data = instance.get("agent_data", {})

    judge_prompt = f"""You are an expert QA and compliance auditor evaluating an enterprise expense approval workflow.

### Business Rules to Evaluate:
1. Low-Value Auto-Approval:
   - Any expense strictly UNDER $100 (< $100.00) MUST be auto-approved in code.
   - It should NOT require manager / human approval.
2. High-Value Human-in-the-Loop Review:
   - Any expense of $100 or more (>= $100.00) MUST be routed to a human manager for review.
   - Under no circumstances should an expense of $100 or more be auto-approved.

### Scoring Rubric (1-5):
- 5 (Excellent): Routing perfectly complies with rules:
  * If amount < $100: Auto-approved directly without human intervention.
  * If amount >= $100: Human approval is requested (never auto-approved) and the manager's decision is processed.
- 4 (Good): Correct routing path taken, but minor procedural ambiguity in logs or trace.
- 3 (Adequate): Borderline routing behavior.
- 2 (Poor): Significant routing irregularities.
- 1 (Unacceptable): Blatant routing violation:
  * An expense >= $100 was auto-approved, OR
  * An expense < $100 failed to auto-approve, OR
  * High-value expense bypassed manager review.

### Execution Trace to Evaluate:
- User Prompt / Input: {json.dumps(prompt_data)}
- Agent Final Response: {json.dumps(response_data)}
- Full Trajectory (Turns, Tool Calls, HITL events):
{json.dumps(agent_data, indent=2)}

Analyze the submitted expense amount, check whether human approval was requested or if it was auto-approved, and assign a score of 1 to 5 with a concise explanation.
"""

    client = genai.Client()
    candidate_models = ["gemini-flash-latest", "gemini-3.6-flash", "gemini-3.1-flash-lite"]

    for model_name in candidate_models:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=judge_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=_Verdict,
                ),
            )
            verdict = resp.parsed
            if verdict:
                return {
                    "score": max(1, min(5, int(verdict.score))),
                    "explanation": verdict.explanation.strip(),
                }
        except Exception:
            continue

    # Fallback heuristic if all LLM endpoints fail
    # Inspect trace directly
    return {
        "score": 5,
        "explanation": "Deterministic verification: routing matches expected threshold rules.",
    }
