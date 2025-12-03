import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

from src.lang_change_utils import build_chat_model

load_dotenv()

MAP_TRANSLATE_PROMPT_DEFAULT = (
    "You are a professional translator. Translate the following text into {target_language}. "
    "Preserve meaning, tone, and as much formatting as practical. Reply with only the translation."
)

REDUCE_TRANSLATE_PROMPT_DEFAULT = (
    "You are consolidating translated chunks into one fluent translation in {target_language}. "
    "Merge the pieces, fix broken sentences, and ensure coherence. Reply with only the final translation."
)


def load_text() -> str:
    """
    Load source text from a file path specified via TRANSLATE_SOURCE_FILE (defaults to data/to_translate.txt).
    """
    source_path = Path(os.getenv("TRANSLATE_SOURCE_FILE", "data/to_translate.txt"))
    print(f"Loading document from: {source_path}")
    if not source_path.exists():
        print(f"File not found: {source_path}. Set TRANSLATE_SOURCE_FILE or place the file there.")
        sys.exit(1)
    try:
        return source_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Error reading {source_path}: {exc}")
        sys.exit(1)


def translate_text(
    text: str,
    *,
    target_language: str = "Spanish",
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    map_prompt_text: str = MAP_TRANSLATE_PROMPT_DEFAULT,
    reduce_prompt_text: str = REDUCE_TRANSLATE_PROMPT_DEFAULT,
    
) -> str:
    """
    Translate text using a chunked map-reduce strategy to keep token sizes manageable.
    """
    llm = build_chat_model(provider=provider, model=model, temperature=temperature)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = text_splitter.split_text(text)

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", map_prompt_text),
            ("human", "{chunk}"),
        ]
    )
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", reduce_prompt_text),
            ("human", "{summaries}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    map_inputs = [{"chunk": chunk, "target_language": target_language} for chunk in split_docs]
    map_summaries = map_chain.batch(map_inputs, config={"max_concurrency": 4})

    joined_summaries = "\n\n".join(map_summaries)
    return reduce_chain.invoke({"summaries": joined_summaries, "target_language": target_language})
