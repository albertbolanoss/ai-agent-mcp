import os
from typing import Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.env_vars_utils import get_ollama_base_url

def build_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """
    Build a chat model for the requested provider. Supports Ollama and OpenAI.
    """
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
