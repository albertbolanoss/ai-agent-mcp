import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter

MAP_PROMPT_DEFAULT = (
    "You are a helpful assistant. Summarize the main themes of this chunk:\n"
    "{chunk}\n\nSummary:"
)
REDUCE_PROMPT_DEFAULT = (
    "You are a helpful assistant. Merge the following partial summaries into a concise "
    "final summary in Spanish:\n{summaries}\n\nFinal summary:"
)

# Load environment variables early
load_dotenv()

def get_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def get_env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

def build_llama_model() -> "OllamaLLM":
    """Init Llama LLM model from localhost."""
    try:
        from langchain_ollama import OllamaLLM
    except ImportError as exc:
        print(f"Cannot import ChatOpenAI. Install langchain-openai. Details: {exc}")
        sys.exit(1)

    return OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=get_env_float("LLM_TEMPERATURE", 0.1),
    )


def build_openai_model() -> object:
    """
    Init an OpenAI chat model compatible with the LCEL chains used below.
    Requires langchain-openai and OPENAI_API_KEY.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        print(f"Cannot import ChatOpenAI. Install langchain-openai. Details: {exc}")
        sys.exit(1)

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=get_env_float("LLM_TEMPERATURE", 0.1),
    )


def load_text_to_summarize() -> str:
    """
    Load source text from a file.
    Path can be set via SUMMARY_SOURCE_FILE env var (defaults to data/to_summarize.txt).
    """
    source_path = Path(os.getenv("SUMMARY_SOURCE_FILE", "data/to_summarize.txt"))
    print(f"Loading document from: {source_path}")
    if not source_path.exists():
        print(f"File not found: {source_path}. Set SUMMARY_SOURCE_FILE or place the file there.")
        sys.exit(1)
    try:
        return source_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Error reading {source_path}: {exc}")
        sys.exit(1)


def main() -> None:
    # ---------------------------------------------------------
    # 1. Get LLM Model
    # ---------------------------------------------------------
    llm = build_llama_model()

    # ---------------------------------------------------------
    # 2. Get Full Document
    # ---------------------------------------------------------
    text_content = load_text_to_summarize()
    docs = [Document(page_content=text_content)]

    # ---------------------------------------------------------
    # 3. CHUNKING
    # ---------------------------------------------------------
    chunk_size = get_env_int("CHUNK_SIZE", 1000)
    chunk_overlap = get_env_int("CHUNK_OVERLAP", 0)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Original document split into {len(split_docs)} chunks.")

    # ---------------------------------------------------------
    # 4. MAP AND REDUCE WITH LCEL
    # ---------------------------------------------------------
    map_prompt_text = os.getenv("MAP_PROMPT", MAP_PROMPT_DEFAULT)
    reduce_prompt_text = os.getenv("REDUCE_PROMPT", REDUCE_PROMPT_DEFAULT)

    map_prompt = PromptTemplate.from_template(map_prompt_text)
    reduce_prompt = PromptTemplate.from_template(reduce_prompt_text)

    map_chain = map_prompt | llm | StrOutputParser()
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    # MAP: summarize each chunk in parallel
    map_inputs = [{"chunk": d.page_content} for d in split_docs]
    map_summaries = map_chain.batch(map_inputs, config={"max_concurrency": 4})

    # REDUCE: consolidate summaries
    joined_summaries = "\n\n".join(map_summaries)
    final_summary = reduce_chain.invoke({"summaries": joined_summaries})

    # ---------------------------------------------------------
    # 5. RESULTS
    # ---------------------------------------------------------
    print("\nFinal Results:")
    print("-" * 20)
    print(final_summary)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
