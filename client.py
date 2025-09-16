# client.py
import asyncio, os, sys
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

async def main():
    # Usa el intérprete actual del venv y desactiva el buffering con -u
    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", "server.py"],
        env=os.environ.copy(),   # hereda OLLAMA_HOST y el venv
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("🔧 Tools:", [t.name for t in tools.tools])

            res_models = await session.call_tool("list_models", {})
            block = res_models.content[0]
            print("📦 Modelos:", block.text if isinstance(block, types.TextContent) else res_models.content)

            res_chat = await session.call_tool("llm_chat", {
                "user": "En una frase: ¿qué es MCP?",
                "model": "llama3.2",
                "system": "Eres un asistente técnico, conciso y claro."
            })
            block = res_chat.content[0]
            print("🤖 Respuesta:", block.text if isinstance(block, types.TextContent) else res_chat.content)

if __name__ == "__main__":
    asyncio.run(main())
