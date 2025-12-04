import asyncio
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from mcp import types
from pydantic import BaseModel, Field

from src.env_vars_utils import get_llm_server
from src.lang_change_utils import stream_chat_completion
from src.mcp_client import MCPClientManager
from src.query_agent import decide_tool

load_dotenv()

app = FastAPI(title="MCP Chat Web", version="0.3.0")
BASE_DIR = Path(__file__).resolve().parent
HOME_TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"


mcp_manager = MCPClientManager()
LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE = get_llm_server()


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Chat history.")
    target_language: Optional[str] = Field(None, description="Target language for translation.")
    translation_text: Optional[str] = Field(None, description="Optional explicit text to translate.")
    summary_text: Optional[str] = Field(
        None,
        description="Deprecated alias for translation_text.",
    )
    chunk_size: int = Field(1000, description="Only for translate tool.")
    chunk_overlap: int = Field(0, description="Only for translate tool.")
    session_id: Optional[str] = Field(
        None,
        description="Optional session identifier to retain a limited message history server-side.",
    )
    max_context_messages: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_MESSAGES", "10")),
        description="Maximum messages to keep per session.",
    )


class SessionMemory:
    """
    Minimal in-memory session store to keep a small rolling history per session_id.
    Not suitable for production multi-instance deployments.
    """

    def __init__(self) -> None:
        self._store: dict[str, List[Message]] = {}
        self._lock = asyncio.Lock()

    async def merge(
        self, session_id: Optional[str], new_messages: List[Message], limit: int
    ) -> List[Message]:
        if not session_id:
            return new_messages

        async with self._lock:
            history = self._store.get(session_id, [])
            combined = history + new_messages
            trimmed = combined[-limit:]
            self._store[session_id] = trimmed
            return trimmed


session_memory = SessionMemory()


def extract_text_content(result: types.CallToolResult) -> str:
    """
    Extract text from MCP result, returning the first text block.
    """
    block = result.content[0]
    if isinstance(block, types.TextContent):
        return block.text
    return str(block)


async def ensure_session_started() -> None:
    try:
        await mcp_manager.start()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not start MCP: {exc}") from exc


def _stream_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_llm_chat(arguments: dict[str, Any]) -> AsyncIterator[str]:
    """
    Stream LLM responses token by token for chat conversations.
    """
    messages = []
    system = arguments.get("system")
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": arguments.get("user", "")})

    async for token in stream_chat_completion(
        messages,
        provider=LLM_PROVIDER,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    ):
        if token:
            yield token


@lru_cache(maxsize=1)
def load_home_template() -> str:
    try:
        return HOME_TEMPLATE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="Home template not found") from exc


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    """
    Simple web UI for interacting with the /chat endpoint.
    """
    return HTMLResponse(content=load_home_template())


@app.on_event("startup")
async def _startup() -> None:
    await ensure_session_started()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await mcp_manager.stop()


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    ChatGPT-style endpoint. The agent chooses between chat (llm_chat) or translation (translate).
    Model/provider are resolved from environment and are not user-controlled.
    """
    await ensure_session_started()

    merged_messages = await session_memory.merge(request.session_id, request.messages, request.max_context_messages)
    
    user_intent = decide_tool([m.model_dump() for m in merged_messages])
    
    if user_intent["isCompleted"] == "true":
        try:
            result = await mcp_manager.call_tool(user_intent["tool"], user_intent["arguments"])
            user_intent = {**user_intent, "reply": extract_text_content(result)}

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error calling tool {user_intent['tool']}: {exc}") from exc
    
    return user_intent


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming version of /chat. Emits events as SSE-style "data: {...}\\n\\n".
    - For llm_chat, tokens are streamed reactively (works with OpenAI and Ollama).
    - For translate, emits a start and a final complete event.
    """
    await ensure_session_started()

    merged_messages = await session_memory.merge(request.session_id, request.messages, request.max_context_messages)
    user_intent = decide_tool([m.model_dump() for m in merged_messages])

    async def event_generator() -> AsyncIterator[str]:
        if user_intent["isCompleted"] != "true":
            yield _stream_event({"type": "router_reply", "reply": user_intent["reply"]})
            return

        tool = user_intent["tool"]
        arguments = user_intent["arguments"]

        yield _stream_event({"type": "start", "tool": tool})

        if tool == "llm_chat":
            transcript: list[str] = []
            async for token in _stream_llm_chat(arguments):
                transcript.append(token)
                yield _stream_event({"type": "token", "delta": token})

            yield _stream_event({"type": "complete", "tool": tool, "reply": "".join(transcript)})
            return

        try:
            result = await mcp_manager.call_tool(tool, arguments)
            reply_text = extract_text_content(result)
            yield _stream_event({"type": "complete", "tool": tool, "reply": reply_text})
        except Exception as exc:
            yield _stream_event(
                {
                    "type": "error",
                    "tool": tool,
                    "message": f"Error calling tool {tool}: {exc}",
                }
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("client:app", host="0.0.0.0", port=8000, reload=False)
