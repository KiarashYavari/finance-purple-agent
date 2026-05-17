# src/messenger.py
"""
Communication utilities for the Finance Purple/White Agent.

Responsibilities:
- Connect to MCP server over SSE.
- Discover MCP tools.
- Call MCP tools.
- Hide MCP transport details from agent.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from mcp import ClientSession
from mcp.client.sse import sse_client


@dataclass
class MCPConnectionConfig:
    """MCP SSE connection configuration."""

    mcp_url: str
    timeout: float = 600.0

    @property
    def sse_url(self) -> str:
        """Normalize MCP URL to the /sse endpoint."""
        if self.mcp_url.endswith("/sse"):
            return self.mcp_url

        match = re.match(r"http://([^:/]+):(\d+)", self.mcp_url)
        if not match:
            raise ValueError(f"Invalid MCP URL: {self.mcp_url}")

        host = match.group(1)
        port = int(match.group(2))

        return f"http://{host}:{port}/sse"


class MCPToolClient:
    """
    Async MCP SSE client.

    Usage:
        async with MCPToolClient(mcp_url) as client:
            tools = await client.list_tools()
            result = await client.call_tool("tool_name", {"x": 1})
    """

    def __init__(self, mcp_url: str, timeout: float = 600.0):
        self.config = MCPConnectionConfig(mcp_url=mcp_url, timeout=timeout)
        self._sse_context = None
        self._session_context = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "MCPToolClient":
        print(f"[PURPLE][MCP] Connecting to {self.config.sse_url}")

        self._sse_context = sse_client(self.config.sse_url, timeout=self.config.timeout)
        read, write = await self._sse_context.__aenter__()

        self._session_context = ClientSession(read, write)
        self._session = await self._session_context.__aenter__()

        await self._session.initialize()

        print("[PURPLE][MCP] Initialized")

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session_context:
            await self._session_context.__aexit__(exc_type, exc, tb)

        if self._sse_context:
            await self._sse_context.__aexit__(exc_type, exc, tb)

        self._session = None

    async def list_tools(self) -> Sequence[Any]:
        """List available MCP tools."""
        session = self._require_session()
        tools_result = await session.list_tools()
        return tools_result.tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call one MCP tool."""
        session = self._require_session()
        return await session.call_tool(tool_name, arguments=arguments)

    def _require_session(self) -> ClientSession:
        if not self._session:
            raise RuntimeError("MCPToolClient is not connected. Use it as an async context manager.")

        return self._session