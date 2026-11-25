"""Agent LangChain client du serveur MCP (transport streamable-http).

1) lancer le serveur :  uv run partie_C_mcp/mcp_server.py
2) lancer l'agent    :  uv run partie_C_mcp/agent_mcp.py      (taper « exit » pour quitter)
"""
import asyncio

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

mcp_client = MultiServerMCPClient(
    {
        "mcp_server": {
            "transport": "streamable_http",
            "url": "http://localhost:24000/mcp",
        }
    }
)


async def main():
    tools = await mcp_client.get_tools()  # découverte des tools exposés par le serveur MCP
    print("Tools MCP :", [t.name for t in tools])
    llm = ChatOllama(model="qwen3-vl:8b", temperature=0, reasoning=False, num_ctx=16384)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="Answer the user question using the provided tools. Answer in French.",
    )
    while True:
        user_query = input("Question : ")
        if user_query == "exit":
            break
        response = await agent.ainvoke({"messages": [HumanMessage(user_query)]})
        print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
