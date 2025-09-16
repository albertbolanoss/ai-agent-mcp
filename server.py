# server.py
from mcp.server.fastmcp import FastMCP
import ollama
import traceback, sys

mcp = FastMCP("OllamaBridge")

@mcp.tool()
def list_models() -> list[str]:
    """Lista modelos disponibles en el daemon de Ollama."""
    data = ollama.list()  # {'models': [{ 'model': 'llama3.2', ...}, ...]}
    return [m["model"] for m in data.get("models", [])]

@mcp.tool()
def llm_chat(
    user: str,
    model: str = "llama3.2",
    system: str | None = None,
    keep_alive: str | int | None = "5m",
) -> str:
    """Envía un chat a un modelo de Ollama y devuelve el texto de respuesta."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    resp = ollama.chat(model=model, messages=messages, keep_alive=keep_alive)
    return resp["message"]["content"]

if __name__ == "__main__":
    # Si algo falla, muestra el traceback en STDERR para que lo veas desde el cliente.
    try:
        mcp.run()  # stdio
    except Exception:
        traceback.print_exc()
        sys.exit(1)
