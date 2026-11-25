"""Serveur MCP (Model Context Protocol) exposant deux tools via le transport streamable-http.

Lancement (depuis la racine TP5/) :
    uv run partie_C_mcp/mcp_server.py
Endpoint : http://localhost:24000/mcp
"""
from pathlib import Path

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from mcp.server.fastmcp import FastMCP

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

# host 0.0.0.0 : le serveur est aussi joignable depuis un conteneur Docker (n8n)
mcp = FastMCP(name="mcp-server", host="0.0.0.0", port=24000)

web_search_client = TavilySearch(max_results=3)


@mcp.tool()
def get_employee_infos(name: str) -> dict:
    """Get information about the given employee (name, salary, seniority)."""
    print(f"[MCP] get_employee_infos({name!r})")
    return {"name": name, "salary": 23000, "seniority": 12}


@mcp.tool()
def search(query: str) -> dict:
    """Search the web for the given query (Tavily)."""
    print(f"[MCP] search({query!r})")
    return web_search_client.invoke({"query": query})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
