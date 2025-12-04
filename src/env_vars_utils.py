import os

from dotenv import load_dotenv

load_dotenv()


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def get_query_agent() -> tuple[str, str, float]:
    """
    Return model name with safe default if env var is missing or blank.
    """
    provider = os.getenv("QUERY_AGENT_PROVIDER", "ollama").strip()
    model = os.getenv("QUERY_AGENT_MODEL", "llama3.2").strip()
    temperature = _env_float("QUERY_AGENT_TEMPERATURE", 0.1)
    return (provider, model, temperature)


def get_llm_server() -> tuple[str, str, float]:
    """
    Return model name with safe default if env var is missing or blank.
    """
    provider = os.getenv("LLM_SERVER_PROVIDER", "ollama").strip()
    model = os.getenv("LLM_SERVER_MODEL", "llama3.2").strip()
    temperature = _env_float("LLM_SERVER_TEMPERATURE", 0.7)
    return (provider, model, temperature)

def get_ollama_base_url() -> str:
    """
    Return Ollama base URL with safe default if env var is missing or blank.
    """
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()

def get_mcp_max_concurrency() -> int:
    """
    Return MCP max concurrency with safe default if env var is missing or invalid.
    """
    return _env_int("MCP_MAX_CONCURRENCY", 4)