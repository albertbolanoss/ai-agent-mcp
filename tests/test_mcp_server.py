import types

import src.mcp_server as server


def test_list_models(monkeypatch):
    monkeypatch.setattr(server.ollama, "list", lambda: {"models": [{"model": "a"}, {"model": "b"}]})
    assert server.list_models() == ["a", "b"]


def test_llm_chat(monkeypatch):
    captured = {}

    def fake_chat_completion(messages, **kwargs):
        captured["messages"] = messages
        return "hello"

    monkeypatch.setattr(server, "chat_completion", fake_chat_completion)

    result = server.llm_chat("hi", system="sys")
    assert result == "hello"
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["role"] == "user"


def test_summarize(monkeypatch):
    called = {}

    def fake_summarize(text, **kwargs):
        called.update({"text": text, "kwargs": kwargs})
        return "summary"

    monkeypatch.setattr(server, "summarize_text", fake_summarize)

    result = server.summarize("long text", chunk_size=10, chunk_overlap=1)
    assert result == "summary"
    assert called["text"] == "long text"
    assert called["kwargs"]["chunk_size"] == 10
    assert called["kwargs"]["chunk_overlap"] == 1
