import types
import pytest

import src.mcp_server as server


def test_main_block_handles_exception(monkeypatch):
    calls = {}

    def boom():
        calls["run"] = True
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "mcp", types.SimpleNamespace(run=boom))
    monkeypatch.setattr(server.traceback, "print_exc", lambda: calls.setdefault("trace", True))

    with pytest.raises(SystemExit) as excinfo:
        try:
            server.mcp.run()
        except Exception:
            server.traceback.print_exc()
            server.sys.exit(1)

    assert calls.get("run") is True
    assert calls.get("trace") is True
    assert excinfo.value.code == 1
