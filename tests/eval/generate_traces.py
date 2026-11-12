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

"""Trace generator for ADK 2.0 expense approval agent evaluations.

Executes evaluation cases from an evaluation dataset through the ADK workflow runner,
intercepts human-in-the-loop (HITL) approval steps, automates approval/rejection decisions,
and serializes full conversation traces to artifacts/traces/generated_traces.json.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from vertexai._genai.types.common import (
    EvalCase,
    EvaluationDataset,
    ResponseCandidate,
)
from vertexai._genai.types.evals import (
    AgentConfig,
    AgentData,
    AgentEvent,
    ConversationTurn,
)

from expense_agent.agent import root_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_interrupt_function_call(event: Any) -> genai_types.FunctionCall | None:
    """Find adk_request_input function call part if present in event content."""
    if not event or not getattr(event, "content", None):
        return None
    content = event.content
    if not hasattr(content, "parts") or not content.parts:
        return None
    for part in content.parts:
        if getattr(part, "function_call", None) and part.function_call.name == "adk_request_input":
            return part.function_call
    return None


async def run_scenario_with_retry(
    eval_case_id: str,
    raw_prompt_text: str,
    max_retries: int = 5,
) -> tuple[list[ConversationTurn], genai_types.Content, genai_types.Content]:
    """Execute a single scenario through Runner with automated HITL decisions and retries."""
    for attempt in range(1, max_retries + 1):
        try:
            session_service = InMemorySessionService()
            user_id = f"eval_user_{eval_case_id}"
            session = await session_service.create_session(user_id=user_id, app_name="app")
            runner = Runner(agent=root_agent, session_service=session_service, app_name="app")

            prompt_content = genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=raw_prompt_text)],
            )

            # Turn 0: Initial submission
            turn_0_events: list[AgentEvent] = [
                AgentEvent(author="user", content=prompt_content)
            ]

            raw_events_turn_0: list[Any] = []
            async for evt in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=prompt_content,
            ):
                raw_events_turn_0.append(evt)
                if getattr(evt, "content", None) and evt.content.parts:
                    author = getattr(evt, "author", "expense_processor") or "expense_processor"
                    turn_0_events.append(AgentEvent(author=author, content=evt.content))

            turns: list[ConversationTurn] = [
                ConversationTurn(turn_index=0, events=turn_0_events)
            ]

            # Check if interrupted for HITL approval
            interrupt_call = None
            last_turn_0_evt = raw_events_turn_0[-1] if raw_events_turn_0 else None
            if last_turn_0_evt:
                interrupt_call = find_interrupt_function_call(last_turn_0_evt)

            final_response_content: genai_types.Content | None = None

            if interrupt_call:
                # Inspect payload/state to automate decision
                args = interrupt_call.args or {}
                payload = args.get("payload", {}) if isinstance(args, dict) else {}
                current_session = await session_service.get_session(
                    user_id=user_id, session_id=session.id, app_name="app"
                )
                state = current_session.state if current_session else {}

                is_injection = bool(
                    payload.get("security_alert")
                    or state.get("security_flag")
                )
                decision = "reject" if is_injection else "approve"
                logger.info(
                    f"[{eval_case_id}] HITL interrupt intercepted. Security flag: {is_injection} -> Decision: '{decision}'"
                )

                resume_msg = genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part.from_function_response(
                            name="adk_request_input",
                            response={"decision": decision, "result": decision},
                        )
                    ],
                )
                resume_msg.parts[0].function_response.id = interrupt_call.id

                turn_1_events: list[AgentEvent] = [
                    AgentEvent(author="user", content=resume_msg)
                ]

                async for evt in runner.run_async(
                    user_id=user_id,
                    session_id=session.id,
                    invocation_id=last_turn_0_evt.invocation_id,
                    new_message=resume_msg,
                ):
                    if getattr(evt, "content", None) and evt.content.parts:
                        author = getattr(evt, "author", "expense_processor") or "expense_processor"
                        turn_1_events.append(AgentEvent(author=author, content=evt.content))

                turns.append(ConversationTurn(turn_index=1, events=turn_1_events))

            # Extract final response content
            for turn in reversed(turns):
                for event in reversed(turn.events):
                    if event.author != "user" and event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                final_response_content = event.content
                                break
                    if final_response_content:
                        break
                if final_response_content:
                    break

            if not final_response_content:
                final_response_content = genai_types.Content(
                    role="model",
                    parts=[genai_types.Part.from_text(text="Workflow completed.")],
                )

            return turns, prompt_content, final_response_content

        except Exception as e:
            logger.warning(f"[{eval_case_id}] Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                raise


async def generate_all_traces(
    dataset_path: str = "tests/eval/datasets/basic-dataset.json",
    output_path: str = "artifacts/traces/generated_traces.json",
) -> None:
    """Generate evaluation traces for all cases in the dataset and save to disk."""
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    with open(dataset_file, encoding="utf-8") as f:
        data = json.load(f)

    eval_cases_in = data.get("eval_cases", [])
    logger.info(f"Loaded {len(eval_cases_in)} cases from {dataset_path}")

    evaluated_cases: list[EvalCase] = []

    for case_data in eval_cases_in:
        case_id = case_data.get("eval_case_id", "unnamed_case")
        prompt_dict = case_data.get("prompt", {})
        raw_text = ""
        if "parts" in prompt_dict and prompt_dict["parts"]:
            raw_text = prompt_dict["parts"][0].get("text", "")

        logger.info(f"Executing scenario: {case_id} ...")
        turns, prompt_content, final_response = await run_scenario_with_retry(case_id, raw_text)

        eval_case = EvalCase(
            eval_case_id=case_id,
            prompt=prompt_content,
            responses=[ResponseCandidate(response=final_response)],
            agent_data=AgentData(
                agents={
                    "expense_processor": AgentConfig(agent_id="expense_processor"),
                    "review_agent": AgentConfig(agent_id="review_agent"),
                },
                turns=turns,
            ),
        )
        evaluated_cases.append(eval_case)
        logger.info(f"Completed scenario: {case_id}")

    dataset = EvaluationDataset(eval_cases=evaluated_cases)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(dataset.model_dump_json(exclude_none=True, indent=2))

    logger.info(f"Saved {len(evaluated_cases)} evaluated traces to {output_path}")


def main() -> None:
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "tests/eval/datasets/basic-dataset.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "artifacts/traces/generated_traces.json"
    asyncio.run(generate_all_traces(dataset_path=dataset_path, output_path=output_path))


if __name__ == "__main__":
    main()
