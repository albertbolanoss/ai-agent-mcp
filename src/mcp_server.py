# server.py
import sys
import traceback

import ollama
from mcp.server.fastmcp import FastMCP

from src.env_vars_utils import get_llm_server
from src.lang_change_utils import chat_completion
from src.summarize_text import summarize_text

mcp = FastMCP("OllamaBridge")

LLM_SERVER_PROVIDER, LLM_SERVER_MODEL, LLM_SERVER_TEMPERATURE = get_llm_server()


@mcp.tool()
def list_models() -> list[str]:
    """List models available in the Ollama daemon."""
    data = ollama.list()  # {'models': [{ 'model': 'llama3.2', ...}, ...]}
    return [m["model"] for m in data.get("models", [])]


@mcp.tool()
def llm_chat(
    user: str,
    system: str | None = None,
) -> str:
    """
    Send a chat-style prompt to the configured LLM and return the response text.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return chat_completion(messages, model=LLM_SERVER_MODEL, provider=LLM_SERVER_PROVIDER, temperature=LLM_SERVER_TEMPERATURE)


@mcp.tool()
def summarize(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 0
) -> str:
    """
    Summarize text in its original language using chunked map-reduce.
    """
    return summarize_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model=LLM_SERVER_MODEL,
        provider=LLM_SERVER_PROVIDER,
        temperature=LLM_SERVER_TEMPERATURE,
    )


if __name__ == "__main__":
    # If something fails, print the traceback to STDERR so the client can see it.
    try:
        mcp.run()  # stdio
    except Exception:
        traceback.print_exc()
        sys.exit(1)
