import asyncio
import types

import pytest

import src.mcp_client as mc


class DummySession:
    def __init__(self):
        self.initialized = False
        self.called_with = None

    async def initialize(self):
        self.initialized = True

    async def call_tool(self, name, args):
        self.called_with = (name, args)
        return "ok"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyContext:
    def __init__(self, pair):
        self.pair = pair
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self.pair

    async def __aexit__(self, exc_type, exc, tb):
        self.entered = False
        return False


def test_call_tool_without_start_raises():
    manager = mc.MCPClientManager(2)
    with pytest.raises(RuntimeError):
        asyncio.run(manager.call_tool("x", {}))


@pytest.mark.anyio
async def test_start_and_call_tool(monkeypatch):
    dummy_session = DummySession()

    def fake_stdio_client(params):
        return DummyContext(("r", "w"))

    def fake_client_session(read, write):
        assert read == "r" and write == "w"
        return dummy_session

    monkeypatch.setattr(mc, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(mc, "ClientSession", fake_client_session)

    manager = mc.MCPClientManager(1)
    await manager.start()

    result = await manager.call_tool("tool", {"a": 1})
    assert result == "ok"
    assert dummy_session.called_with == ("tool", {"a": 1})

    await manager.stop()


@pytest.mark.anyio
async def test_start_is_idempotent(monkeypatch):
    dummy_session = DummySession()

    def fake_stdio_client(params):
        return DummyContext(("r", "w"))

    def fake_client_session(read, write):
        return dummy_session

    monkeypatch.setattr(mc, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(mc, "ClientSession", fake_client_session)

    manager = mc.MCPClientManager(1)
    await manager.start()
    # second start should do nothing and not raise
    await manager.start()
    await manager.stop()


@pytest.mark.anyio
async def test_stop_without_start_is_safe():
    manager = mc.MCPClientManager(1)
    # should not raise
    await manager.stop()
