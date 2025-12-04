import os
import tempfile

import pytest

import src.summarize_text as st


def test_load_text_reads_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp.write("hello")
        tmp_path = tmp.name
    try:
        os.environ["SUMMARY_SOURCE_FILE"] = tmp_path
        assert st.load_text() == "hello"
    finally:
        os.unlink(tmp_path)
        os.environ.pop("SUMMARY_SOURCE_FILE", None)

def test_load_text_missing_file(monkeypatch):
    monkeypatch.setenv("SUMMARY_SOURCE_FILE", "does_not_exist.txt")
    with pytest.raises(SystemExit):
        st.load_text()


def test_summarize_text_pipeline(monkeypatch):
    # Minimal fake components to exercise map+reduce
    class FakeSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def split_text(self, text):
            return ["part1", "part2"]

    class FakePrompt:
        def __init__(self, stage: str):
            self.stage = stage

        def __or__(self, other):
            return FakeChain(self.stage)

    class FakePromptFactory:
        @staticmethod
        def from_messages(messages):
            text_repr = " ".join(str(m) for m in messages)
            stage = "summaries" if "summaries" in text_repr else "chunk"
            return FakePrompt(stage)

    class FakeChain:
        def __init__(self, stage: str):
            self.stage = stage

        def __or__(self, other):
            return self

        def batch(self, inputs, config=None):
            return [f"summary:{item['chunk']}" for item in inputs]

        def invoke(self, data):
            return f"FINAL:{data['summaries']}"

    monkeypatch.setattr(st, "CharacterTextSplitter", type("S", (), {"from_tiktoken_encoder": staticmethod(lambda **kwargs: FakeSplitter())}))
    monkeypatch.setattr(st, "ChatPromptTemplate", FakePromptFactory)
    monkeypatch.setattr(st, "StrOutputParser", lambda: None)
    monkeypatch.setattr(st, "build_chat_model", lambda **kwargs: object())

    result = st.summarize_text("dummy", chunk_size=10, chunk_overlap=2)
    assert result == "FINAL:summary:part1\n\nsummary:part2"
