import json

import pytest

import src.query_agent as qa


def test_decide_tool_invalid_json(monkeypatch):
    # Force invalid JSON to hit error path and base response
    monkeypatch.setattr(qa, "chat_completion", lambda *args, **kwargs: "not json")
    result = qa.decide_tool([{ "role": "user", "content": "hola" }])
    assert result["tool"] is None
    assert result["isCompleted"] == "false"
    assert "JSON" in result["reply"] or "Devuelve" in (result["reply"] or "")


def test_parse_decision_with_fences(monkeypatch):
    response = "```\n{\"tool\":\"llm_chat\",\"arguments\":{}}\n```"
    monkeypatch.setattr(qa, "chat_completion", lambda *args, **kwargs: response)
    result = qa.decide_tool([{ "role": "user", "content": "hola" }])
    assert result["tool"] == "llm_chat"
    assert result["arguments"]["user"] == "hola"


def test_parse_decision_snippet(monkeypatch):
    # JSON embedded in extra text should be extracted
    raw = "noise {\"tool\":\"summarize\",\"arguments\":{\"text\":\"x\"}} trailing"
    monkeypatch.setattr(qa, "chat_completion", lambda *args, **kwargs: raw)
    res = qa.decide_tool([{ "role": "user", "content": "hola" }])
    assert res["tool"] == "summarize"
