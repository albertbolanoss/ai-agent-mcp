import types
import asyncio
import sys

import pytest

import src.lang_change_utils as lc


class DummyChunk:
    def __init__(self, content):
        self.content = content


class DummyChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.invoked_with = None

    def invoke(self, messages):
        self.invoked_with = messages
        return types.SimpleNamespace(content="done")

    async def astream(self, messages):
        for item in [DummyChunk("A"), DummyChunk([{"text": "B"}, types.SimpleNamespace(text="C")])]:
            yield item


def make_dummy_module(name, cls_name="ChatOllama"):
    mod = types.ModuleType(name)
    setattr(mod, cls_name, DummyChatModel)
    sys.modules[name] = mod
    return mod


def test_build_chat_model_ollama(monkeypatch):
    make_dummy_module("langchain_ollama")
    monkeypatch.setattr(lc, "get_ollama_base_url", lambda: "http://fake")

    model = lc.build_chat_model(provider="ollama", model="llama", temperature=0.2, streaming=True)

    assert isinstance(model, DummyChatModel)
    assert model.kwargs["model"] == "llama"
    assert model.kwargs["base_url"] == "http://fake"
    assert model.kwargs["streaming"] is True


def test_build_chat_model_openai(monkeypatch):
    make_dummy_module("langchain_openai", "ChatOpenAI")
    model = lc.build_chat_model(provider="openai", model="gpt", temperature=0.5, streaming=False)
    assert isinstance(model, DummyChatModel)
    assert model.kwargs["model"] == "gpt"
    assert model.kwargs["streaming"] is False


def test_build_chat_model_gemini(monkeypatch):
    make_dummy_module("langchain_google_genai", "ChatGoogleGenerativeAI")
    model = lc.build_chat_model(provider="gemini", model="g1", temperature=0.3, streaming=True)
    assert isinstance(model, DummyChatModel)
    assert model.kwargs["model"] == "g1"


def test_build_chat_model_anthropic(monkeypatch):
    make_dummy_module("langchain_anthropic", "ChatAnthropic")
    model = lc.build_chat_model(provider="anthropic", model="claude", temperature=0.4, streaming=False)
    assert isinstance(model, DummyChatModel)
    assert model.kwargs["model"] == "claude"


def test_build_chat_model_unsupported():
    with pytest.raises(ValueError):
        lc.build_chat_model(provider="unknown", model="x")


def test_chat_completion_invokes(monkeypatch):
    dummy = DummyChatModel()
    monkeypatch.setattr(lc, "build_chat_model", lambda **kwargs: dummy)

    result = lc.chat_completion([
        {"role": "user", "content": "hola"},
    ], provider="ollama", model="m", temperature=0.1)

    assert result == "done"
    assert dummy.invoked_with[0].__class__.__name__ == "HumanMessage"
    assert dummy.invoked_with[0].content == "hola"


@pytest.mark.anyio
async def test_stream_chat_completion_stream(monkeypatch):
    dummy = DummyChatModel()
    monkeypatch.setattr(lc, "build_chat_model", lambda **kwargs: dummy)

    outputs = []
    async for token in lc.stream_chat_completion([
        {"role": "user", "content": "hola"},
    ], provider="ollama", model="m", temperature=0.1):
        outputs.append(token)

    assert outputs == ["A", "BC"]


@pytest.mark.anyio
async def test_stream_chat_completion_fallback(monkeypatch):
    monkeypatch.setattr(lc, "build_chat_model", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(lc, "chat_completion", lambda *args, **kwargs: "fallback")

    outputs = []
    async for token in lc.stream_chat_completion([
        {"role": "user", "content": "hola"},
    ], provider="ollama", model="m", temperature=0.1):
        outputs.append(token)

    assert outputs == ["fallback"]


@pytest.mark.anyio
async def test_stream_chat_completion_runtime_error(monkeypatch):
    class BadDummy(DummyChatModel):
        async def astream(self, messages):
            raise RuntimeError("stream fail")

    dummy = BadDummy()
    monkeypatch.setattr(lc, "build_chat_model", lambda **kwargs: dummy)
    monkeypatch.setattr(lc, "chat_completion", lambda *args, **kwargs: "recovered")

    outputs = []
    async for token in lc.stream_chat_completion(
        [{"role": "user", "content": "hola"}],
        provider="ollama",
        model="m",
        temperature=0.1,
    ):
        outputs.append(token)

    assert outputs == ["recovered"]


def test_to_langchain_messages_roles():
    msgs = lc._to_langchain_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "hey"},
        ]
    )
    assert msgs[0].__class__.__name__ == "SystemMessage"
    assert msgs[1].__class__.__name__ == "AIMessage"
    assert msgs[2].__class__.__name__ == "HumanMessage"
