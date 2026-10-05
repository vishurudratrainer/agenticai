import asyncio
import os
import sys
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.abspath("public_api_server.py")],
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            
            # Map tools for Ollama
            tools_for_ai = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                }
            } for t in mcp_tools.tools]

            # Complex query: "What is the weather in Tokyo and who is the prime minister of Japan?"
            messages = [{"role": "user", "content": "Tell me the weather in Paris"}]
            
            print("--- Agent Thinking ---")
            response = ollama.chat(model="mistral", messages=messages, tools=tools_for_ai)

            while response.message.tool_calls:
                messages.append(response.message)
                for call in response.message.tool_calls:
                    print(f"🛠️  Using tool: {call.function.name}")
                    result = await session.call_tool(call.function.name, call.function.arguments)
                    messages.append({"role": "tool", "content": result.content[0].text, "name": call.function.name})
                
                response = ollama.chat(model="mistral", messages=messages, tools=tools_for_ai)

            print(f"\n✨ Final Answer:\n{response.message.content}")

if __name__ == "__main__":
    asyncio.run(main())