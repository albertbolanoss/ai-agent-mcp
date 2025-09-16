# server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DemoServer")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Suma dos números"""
    return a + b

@mcp.tool()
def echo(text: str) -> str:
    """Devuelve el mismo texto"""
    return f"echo: {text}"

# Ejecuta el servidor en modo por defecto (stdio)
if __name__ == "__main__":
    mcp.run()
