"""Serveur MCP du SAV : publie l'outil distant `rag_multimodal` (transport streamable-http).

Lancement (depuis la racine TP5/) :
    uv run partie_D_synthese/serveur_mcp_rag.py
Endpoint : http://localhost:25000/mcp
"""
import anyio
from mcp.server.fastmcp import FastMCP

from rag_multimodal import index_manuels, repondre

mcp = FastMCP(name="mcp-sav-documentation", host="0.0.0.0", port=25000)


@mcp.tool()
async def rag_multimodal(question: str, image_ref: str = "") -> str:
    """Recherche dans les manuels des appareils (passages ET figures/pictogrammes) et produit une réponse
    sourcée (manuel, page, figure). Pour toute question de dépannage ou d'utilisation d'un appareil.
    Args:
        question: la question du client
        image_ref: référence de la photo jointe par le client (ex. "img_px200_01"), vide s'il n'y en a pas
    """
    print(f"[MCP] rag_multimodal(question={question!r}, image_ref={image_ref!r})")
    # l'agent RAG (LLM local) est bloquant : exécution dans un thread pour ne pas bloquer le serveur
    return await anyio.to_thread.run_sync(repondre, question, image_ref)


if __name__ == "__main__":
    index_manuels()  # indexation des manuels au démarrage
    mcp.run(transport="streamable-http")
