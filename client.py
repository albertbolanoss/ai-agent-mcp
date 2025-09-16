# client.py
import asyncio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

async def main():
    # 1) Parámetros del servidor a lanzar por stdio (este mismo repo: server.py)
    params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env=None
    )

    # 2) Conectar por stdio y crear la sesión
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # 3) Inicializar el protocolo MCP
            await session.initialize()

            # 4) Listar herramientas
            tools = await session.list_tools()
            print("🔧 Tools:", [t.name for t in tools.tools])

            # 5) Llamar a la tool 'add'
            res_add = await session.call_tool("add", {"a": 7, "b": 5})
            block = res_add.content[0]
            if isinstance(block, types.TextContent):
                print("add =>", block.text)
            else:
                print("add =>", res_add)

            # 6) Llamar a la tool 'echo'
            res_echo = await session.call_tool("echo", {"text": "Hola MCP!"})
            block = res_echo.content[0]
            if isinstance(block, types.TextContent):
                print("echo =>", block.text)
            else:
                print("echo =>", res_echo)

if __name__ == "__main__":
    asyncio.run(main())
