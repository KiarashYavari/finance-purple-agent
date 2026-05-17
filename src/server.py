# src/server.py
"""
Finance Purple/White Agent server.

This is the only file that knows about:
- A2A server setup
- Agent card
- Uvicorn
- CLI runtime configuration

It should not contain reasoning logic.
"""

from __future__ import annotations

import argparse
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from src.executer import Executer


class FinancePurpleAgentServer:
    """Server wrapper for AgentBeats/A2A compatibility."""

    def __init__(
        self,
        host: str,
        port: int,
        card_url: str | None = None,
    ):
        self.host = host
        self.port = port
        self.card_url = card_url

    def create_agent_card(self) -> AgentCard:
        """Create the A2A agent card."""
        skill = AgentSkill(
            id="finance-question-answering",
            name="Finance Question Answering",
            description=(
                "Answers finance benchmark questions by using MCP tools exposed "
                "by the green agent."
            ),
            tags=[
                "finance",
                "mcp",
                "a2a",
                "tool-calling",
                "benchmark",
            ],
            examples=[
                "What did Apple report as total net sales in fiscal year 2023?",
                "Did Netflix mention advertising revenue growth in its latest filing?",
            ],
        )

        return AgentCard(
            name=os.getenv("PURPLE_AGENT_CARD_NAME", "Finance Purple Agent"),
            description=(
                "A finance benchmark answering agent that receives questions over A2A, "
                "uses MCP tools, and returns concise answers."
            ),
            url=self.card_url or f"http://{self.host}:{self.port}/",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[skill],
        )

    def create_app(self):
        """Create the A2A Starlette application."""
        request_handler = DefaultRequestHandler(
            agent_executor=Executer(),
            task_store=InMemoryTaskStore(),
        )

        a2a_app = A2AStarletteApplication(
            agent_card=self.create_agent_card(),
            http_handler=request_handler,
        )

        return a2a_app.build()

    def run(self) -> None:
        """Run the server."""
        print("[PURPLE] ═══════════════════════════════════")
        print("[PURPLE] Finance Purple Agent")
        print(f"[PURPLE] Server: {self.host}:{self.port}")
        print("[PURPLE] A2A: enabled")
        print("[PURPLE] MCP: external client mode")
        print("[PURPLE] ═══════════════════════════════════")

        uvicorn.run(
            self.create_app(),
            host=self.host,
            port=self.port,
            log_level=os.getenv("LOG_LEVEL", "info"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance Purple Agent")

    parser.add_argument(
        "--host",
        default=os.getenv("PURPLE_AGENT_HOST", os.getenv("WHITE_AGENT_HOST", "0.0.0.0")),
        help="Host to bind the A2A server",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PURPLE_AGENT_PORT", os.getenv("WHITE_AGENT_PORT", "9009"))),
        help="Port for the A2A server",
    )

    parser.add_argument(
        "--card-url",
        type=str,
        default=os.getenv("PURPLE_CARD_URL"),
        help="Public URL where this agent card is served",
    )

    args = parser.parse_args()

    server = FinancePurpleAgentServer(
        host=args.host,
        port=args.port,
        card_url=args.card_url,
    )
    server.run()


if __name__ == "__main__":
    main()