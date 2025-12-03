import asyncio
import os
import sys
from typing import Any, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClientManager:
    """
    Keeps the MCP server process (server.py) alive and exposes tool calls.
    """

    def __init__(self) -> None:
        self._stdio_cm = None
        self._session_cm = None
        self._stdio_pair: tuple[Any, Any] | None = None
        self.session: ClientSession | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        # If already started, do nothing.
        if self.session:
            return

        # Create parameters to launch the MCP server process via stdio.
        params = StdioServerParameters(
            command=sys.executable,
            # Run as a module so imports like `src.*` resolve correctly.
            args=["-u", "-m", "src.mcp_server"],
            env=os.environ.copy(),
        )

        # The MCP server is launched and the communication channels to be used by the MCP session are obtained via stdio.
        self._stdio_cm = stdio_client(params)
        self._stdio_pair = await self._stdio_cm.__aenter__()
        read, write = self._stdio_pair

        # Create and initialize the MCP client session.
        self._session_cm = ClientSession(read, write)
        self.session = await self._session_cm.__aenter__()
        await self.session.initialize()

    async def stop(self) -> None:
        if self._session_cm:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None
            self.session = None

        if self._stdio_cm:
            await self._stdio_cm.__aexit__(None, None, None)
            self._stdio_cm = None
            self._stdio_pair = None

    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        if not self.session:
            raise RuntimeError("MCP session not started")

        async with self._lock:
            return await self.session.call_tool(name, arguments)
