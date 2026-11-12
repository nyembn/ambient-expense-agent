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

"""FastAPI web service for ambient expense-approval agent driven by Pub/Sub events."""

import contextlib
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import google.auth
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from google.adk import Event
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.genai import types

from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes
from app.app_utils.typing import Feedback

load_dotenv()

# Standard Python logging for console logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ambient-expense-agent")

try:
    _, project_id = google.auth.default()
except Exception:
    project_id = None

allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def normalize_subscription(subscription: str | None) -> str:
    """Normalize a fully-qualified Pub/Sub subscription path down to a short name.

    Pub/Sub sends subscription as 'projects/<project>/subscriptions/<sub_name>'.
    Normalizing it down to '<sub_name>' keeps session records clean and readable.
    """
    if not subscription:
        return "pubsub-trigger"
    return subscription.strip().rstrip("/").split("/")[-1] or "pubsub-trigger"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.agent import app as adk_app
    from app.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,  # Enables Playground Dev UI alongside ambient endpoints
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=False,  # Telemetry disabled per developer checklist
    lifespan=lifespan,
)
app.title = "ambient-expense-agent"
app.description = "Ambient API for expense approval workflow driven by Pub/Sub events"


@app.post("/")
@app.post("/pubsub")
@app.post("/apps/app/trigger/pubsub")
async def handle_pubsub_trigger(request: Request) -> dict[str, Any]:
    """Accept Pub/Sub push trigger messages and feed them into the agent workflow."""
    try:
        body = await request.json()
    except Exception as e:
        logger.error("Failed to parse JSON body: %s", e)
        raise HTTPException(status_code=400, detail="Invalid JSON body") from e

    raw_subscription = body.get("subscription") if isinstance(body, dict) else None
    user_id = normalize_subscription(raw_subscription)

    message_text = json.dumps(body)
    session_id = str(uuid.uuid4())
    session_service = services.get_session_service()
    agent_app_name = getattr(app.state, "agent_app_name", "app")

    session = await session_service.create_session(
        app_name=agent_app_name,
        user_id=user_id,
        session_id=session_id,
    )

    new_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=message_text)],
    )

    logger.info(
        "Processing Pub/Sub trigger: subscription=%s (normalized user_id=%s), session_id=%s",
        raw_subscription,
        user_id,
        session.id,
    )

    events: list[Event] = []
    async for event in app.state.runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=new_message,
    ):
        events.append(event)

    return {
        "status": "success",
        "subscription": user_id,
        "session_id": session.id,
        "events_count": len(events),
    }


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback."""
    logger.info("Feedback received: %s", feedback.model_dump())
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
