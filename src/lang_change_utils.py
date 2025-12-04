import os
from typing import Any, AsyncIterator, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.env_vars_utils import get_ollama_base_url

def build_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    streaming: bool | None = None,
) -> BaseChatModel:
    """
    Build a chat model for the requested provider. Supports Ollama, OpenAI, Gemini, and Anthropic.
    """
    # Some providers (e.g., ChatOpenAI) require an explicit bool, not None.
    stream_flag = bool(streaming) if streaming is not None else False

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "langchain-google-genai is required to use provider 'gemini'. Install it first."
            ) from exc

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            streaming=stream_flag,
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "langchain-anthropic is required to use provider 'anthropic'. Install it first."
            ) from exc

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            streaming=stream_flag,
        )

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "langchain-openai is required to use provider 'openai'. Install it first."
            ) from exc

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=stream_flag,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "langchain-ollama is required to use provider 'ollama'. Install it first."
            ) from exc

        return ChatOllama(
            model=model,
            base_url=get_ollama_base_url(),
            temperature=temperature,
            streaming=stream_flag,
        )

    raise ValueError(f"Unsupported provider '{provider}'. Use 'ollama' or 'openai'.")


def _to_langchain_messages(messages: List[dict[str, Any]]) -> List[Any]:
    """
    Convert list[dict] with roles (system/user/assistant) into LangChain message objects.
    """
    converted: List[Any] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def chat_completion(
    messages: List[dict[str, Any]],
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """
    Simple chat completion that works for both Ollama and OpenAI providers.
    """
    chat_model = build_chat_model(provider=provider, model=model, temperature=temperature)
    response = chat_model.invoke(_to_langchain_messages(messages))
    return response.content


async def stream_chat_completion(
    messages: List[dict[str, Any]],
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> AsyncIterator[str]:
    """
    Stream chat completions token-by-token for providers that support it.
    Falls back to yielding the full response if streaming is unavailable.
    """
    try:
        chat_model = build_chat_model(
            provider=provider,
            model=model,
            temperature=temperature,
            streaming=True,
        )
    except Exception:
        # If the provider/client cannot be constructed with streaming, emit a single shot.
        yield chat_completion(messages, provider=provider, model=model, temperature=temperature)
        return

    try:
        async for chunk in chat_model.astream(_to_langchain_messages(messages)):
            text = getattr(chunk, "content", None)
            if not text:
                continue
            if isinstance(text, list):
                # Some providers return a list of blocks; keep only text-like pieces.
                parts = []
                for part in text:
                    if isinstance(part, dict) and "text" in part:
                        parts.append(str(part["text"]))
                    elif hasattr(part, "text"):
                        parts.append(str(getattr(part, "text")))
                    else:
                        parts.append(str(part))
                yield "".join(parts)
            else:
                yield str(text)
    except Exception:
        # As a safety net, fall back to a single non-streaming completion.
        yield chat_completion(messages, provider=provider, model=model, temperature=temperature)
