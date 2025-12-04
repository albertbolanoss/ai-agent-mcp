import json

import pytest

import src.query_agent as qa


def test_decide_tool_summarize_complete(monkeypatch):
    def fake_completion(messages, model, provider, temperature):
        return json.dumps({
            "tool": "summarize",
            "arguments": {"text": "hola", "chunk_size": 500},
            "isCompleted": "true",
            "reply": None,
        })

    monkeypatch.setattr(qa, "chat_completion", fake_completion)

    result = qa.decide_tool([
        {"role": "user", "content": "Resume este texto"},
    ])

    assert result["tool"] == "summarize"
    assert result["isCompleted"] == "true"
    assert result["arguments"]["text"] == "hola"
    assert result["arguments"]["chunk_size"] == 500
    assert result["arguments"]["chunk_overlap"] == 0


def test_decide_tool_summarize_missing(monkeypatch):
    def fake_completion(messages, model, provider, temperature):
        return json.dumps({
            "tool": "summarize",
            "arguments": {},
            "isCompleted": "false",
            "reply": None,
        })

    monkeypatch.setattr(qa, "chat_completion", fake_completion)

    result = qa.decide_tool([
        {"role": "user", "content": "Necesito un resumen"},
    ])

    assert result["tool"] is None
    assert result["isCompleted"] == "false"
    assert "text" in (result["reply"] or "")


def test_decide_tool_llm_chat_fills_defaults(monkeypatch):
    def fake_completion(messages, model, provider, temperature):
        return json.dumps({
            "tool": "llm_chat",
            "arguments": {},
            "isCompleted": "true",
            "reply": None,
        })

    monkeypatch.setattr(qa, "chat_completion", fake_completion)

    history = [
        {"role": "system", "content": "context"},
        {"role": "user", "content": "Hola"},
    ]

    result = qa.decide_tool(history)

    assert result["tool"] == "llm_chat"
    assert result["arguments"]["user"] == "Hola"
    # The system argument should include the history string
    assert "system: context" in result["arguments"].get("system", "")
    assert "user: Hola" in result["arguments"].get("system", "")
