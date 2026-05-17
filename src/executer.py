# src/executer.py
"""
A2A executor bridge for the Finance Purple/White Agent.

Responsibilities:
- Receive an A2A message from server.py.
- Extract question and MCP URL.
- Open MCP tool client.
- Call FinancePurpleAgent business logic.
- Publish result back through TaskUpdater.
"""

from __future__ import annotations

import json
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InvalidRequestError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import get_message_text, new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from src.agent import FinancePurpleAgent
from src.messenger import MCPToolClient


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class Executer(AgentExecutor):
    """
        A2A AgentExecutor implementation.

        One FinancePurpleAgent instance is kept per A2A context_id.
    """

    def __init__(self):
        self.agents: dict[str, FinancePurpleAgent] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle one incoming A2A task."""
        message = context.message

        if not message:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task

        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already processed: {task.status.state}"
                )
            )

        if not task:
            task = new_task(message)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        updater = TaskUpdater(event_queue, task.id, context_id)

        agent = self.agents.get(context_id)
        if not agent:
            agent = FinancePurpleAgent()
            self.agents[context_id] = agent

        await updater.start_work()

        try:
            payload = self._extract_payload(message)
            question = payload.get("question")
            mcp_url = payload.get("mcp_url")

            if not question:
                raise ValueError("Missing required field: question")

            if not mcp_url:
                raise ValueError("Missing required field: mcp_url")

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    "Connecting to MCP tools and analyzing the question...",
                    context_id=context_id,
                    task_id=task.id,
                ),
            )

            async with MCPToolClient(mcp_url=mcp_url) as tool_client:
                answer = await agent.answer_question(
                    question=question,
                    tool_client=tool_client,
                )

            await updater.add_artifact(
                parts=[
                    Part(
                        root=TextPart(
                            kind="text",
                            text=answer,
                        )
                    )
                ],
                name="finance-answer",
            )

            await updater.complete()

        except Exception as exc:
            print(f"[PURPLE][EXECUTER] Task failed: {exc}")
            await updater.failed(
                new_agent_text_message(
                    f"Agent error: {exc}",
                    context_id=context_id,
                    task_id=task.id,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancellation is not supported."""
        raise ServerError(error=UnsupportedOperationError())

    @staticmethod
    def _extract_payload(message) -> dict[str, Any]:
        """
        Extract payload from an A2A Message.

        Supported input shapes:
        1. JSON text:
            {"question": "...", "mcp_url": "http://green:9001"}

        2. Legacy text fallback:
            plain question string
        """
        text = get_message_text(message)

        if not text:
            return {}

        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        return {"question": text}