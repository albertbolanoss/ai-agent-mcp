import os

import src.env_vars_utils as env


def test_get_query_agent_defaults(monkeypatch):
    monkeypatch.delenv("QUERY_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("QUERY_AGENT_MODEL", raising=False)
    monkeypatch.delenv("QUERY_AGENT_TEMPERATURE", raising=False)

    provider, model, temperature = env.get_query_agent()

    assert provider == "ollama"
    assert model == "llama3.2"
    assert temperature == 0.1


def test_get_query_agent_env(monkeypatch):
    monkeypatch.setenv("QUERY_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("QUERY_AGENT_MODEL", "gpt-4o")
    monkeypatch.setenv("QUERY_AGENT_TEMPERATURE", "0.7")

    provider, model, temperature = env.get_query_agent()

    assert provider == "openai"
    assert model == "gpt-4o"
    assert temperature == 0.7


def test_get_llm_server_invalid_numbers(monkeypatch):
    monkeypatch.setenv("LLM_SERVER_TEMPERATURE", "not-a-number")
    monkeypatch.setenv("LLM_SERVER_MODEL", "foo")
    monkeypatch.setenv("LLM_SERVER_PROVIDER", "bar")

    provider, model, temperature = env.get_llm_server()

    assert provider == "bar"
    assert model == "foo"
    assert temperature == 0.7  # fallback default


def test_get_mcp_max_concurrency(monkeypatch):
    monkeypatch.setenv("MCP_MAX_CONCURRENCY", "8")
    assert env.get_mcp_max_concurrency() == 8

    monkeypatch.setenv("MCP_MAX_CONCURRENCY", "bad")
    assert env.get_mcp_max_concurrency() == 4


def test_get_ollama_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.com")
    assert env.get_ollama_base_url() == "http://example.com"
