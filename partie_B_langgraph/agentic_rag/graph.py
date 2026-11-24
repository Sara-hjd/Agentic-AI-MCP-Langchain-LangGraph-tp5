"""Chatbot Agentic RAG avec LangGraph.

    START → agent ──(tool_calls ?)──► tools ─┐
              ▲  └──(réponse finale)──► END   │
              └──────────────────────────────┘

- nœud `agent` (MIND) : le LLM décide s'il doit chercher dans le cours, reformule la requête, puis répond ;
- nœud `tools` (BODY) : exécute `retriever_tool` (recherche hybride CLIP + BM25) ;
- la boucle tools → agent permet plusieurs recherches successives (pattern ReAct).

`graph` est exporté sans checkpointer : LangGraph Studio (`langgraph dev`) fournit sa propre persistance.
`build_graph(checkpointer)` sert aux autres usages (notebook, Streamlit).
"""
import json
import logging
import uuid
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain.messages import AIMessage, SystemMessage
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agentic_rag.rag import base_connaissances

load_dotenv(override=True)
logger = logging.getLogger("agentic_rag")  # journal des décisions de l'agent (comme dans le cours)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# ----------------------------------------------------------------------------- Tool
@tool
def retriever_tool(query: str) -> str:
    """Recherche dans le support de cours « Agentic AI et MCP » du Pr. Youssfi : agents (MIND + BODY),
    pattern ReAct, LangChain, LangGraph (state, nœuds, mémoire), MCP (architecture, transports,
    serveur et client MCP), activité pratique SAV.
    Args:
        query: requête de recherche, avec des mots-clés en français ET en anglais
    Returns:
        les passages trouvés, préfixés par leur numéro de page
    """
    try:
        docs = base_connaissances().recherche(query)
        return "\n\n".join(f"[page {d.metadata['page']}] {d.page_content}" for d in docs)
    except Exception as e:
        return f"Erreur lors de la recherche : {e}"


tools = [retriever_tool]

# ----------------------------------------------------------------------------- LLM
# qwen3-vl:8b via Ollama (modèle « thinking »). reasoning=True : le raisonnement est renvoyé à part
# (additional_kwargs["reasoning_content"]), ce qui permet de corriger le bug de parsing ci-dessous.
# num_predict : plafond de tokens générés (raisonnement compris) pour couper une réflexion qui s'emballe.
llm = ChatOllama(model="qwen3-vl:8b", temperature=0, reasoning=True, num_ctx=16384, num_predict=2048)
llm_with_tools = llm.bind_tools(tools)
# Modèle de secours (température > 0) pour les nouvelles tentatives : à température 0, la génération est
# déterministe et une réflexion qui a échoué se reproduirait à l'identique.
llm_retry_with_tools = ChatOllama(model="qwen3-vl:8b", temperature=0.6, reasoning=True, num_ctx=16384,
                                  num_predict=2048).bind_tools(tools)

SYSTEM_PROMPT = """Tu es un assistant pédagogique spécialisé dans le cours « Agentic AI et MCP ».
- Pour toute question sur le cours, appelle retriever_tool. Le support mélange le français et l'anglais :
  formule la requête avec des mots-clés en français ET en anglais (ex. « transport MCP transport mechanisms protocol »).
- Si les passages trouvés ne suffisent pas, tu peux relancer une recherche avec une autre formulation.
- Chaque nouvelle question sur un sujet technique demande une NOUVELLE recherche, même si une recherche a déjà été faite plus tôt dans la conversation.
- Réponds uniquement à partir des passages trouvés, en reprenant toutes les informations pertinentes (listes complètes), et cite les pages (ex. [page 11]).
- Pour une salutation ou une question sur la conversation elle-même (ex. le prénom de l'utilisateur), réponds directement sans outil.
- N'affirme jamais qu'une information est absente du cours sans avoir d'abord appelé retriever_tool. Si, après recherche,
  la réponse ne se trouve pas dans les passages, réponds : « Je ne sais pas — cette information n'est pas dans le support de cours. »
Réponds en français, de façon concise."""


# ----------------------------------------------------------------------------- Graphe
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def tool_call_dans_raisonnement(response) -> AIMessage | None:
    """Contournement d'un bug du parser Ollama « qwen3-vl-thinking » : le modèle écrit parfois son appel
    d'outil ({"name": ..., "arguments": ...}) à l'intérieur du bloc de raisonnement ; la réponse arrive alors
    vide, sans tool_calls. On récupère cet appel et on le transforme en véritable tool_call."""
    texte = response.additional_kwargs.get("reasoning_content", "")
    debut = texte.find('{"name"')
    if debut < 0:
        return None
    try:
        appel, _ = json.JSONDecoder().raw_decode(texte[debut:])
    except json.JSONDecodeError:
        return None
    if appel.get("name") not in {t.name for t in tools}:
        return None
    return AIMessage(content="", tool_calls=[{"name": appel["name"], "args": appel.get("arguments", {}),
                                              "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}])


def agent_node(state: AgentState) -> dict:
    """Nœud agent : appelle le LLM (avec tools) sur le system prompt + l'historique."""
    logger.info("AGENT NODE : appel du LLM avec %d message(s)", len(state["messages"]))
    messages = [SystemMessage(SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    # Garde-fous : le modèle « thinking » renvoie parfois une réponse vide
    for tentative in range(3):
        if response.content.strip() or response.tool_calls:
            break
        if (recupere := tool_call_dans_raisonnement(response)) is not None:
            logger.info("   ⚠ appel d'outil trouvé dans le raisonnement : récupéré")
            response = recupere
            break
        logger.info("   ⚠ réponse vide, nouvelle tentative %d", tentative + 1)
        response = llm_retry_with_tools.invoke(messages + [SystemMessage(
            "Ta réponse précédente était vide. Réponds maintenant à la dernière question de l'utilisateur "
            "(appelle retriever_tool si elle porte sur le cours).")])
    if response.tool_calls:
        for tc in response.tool_calls:
            logger.info("   ✓ décision : %s(%s)", tc["name"], tc["args"])
    else:
        logger.info("   ✓ réponse finale générée")
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Routage : vers les tools s'il y a des appels d'outils, sinon fin."""
    if state["messages"][-1].tool_calls:
        logger.info("🔀 ROUTAGE → tools")
        return "tools"
    logger.info("🔀 ROUTAGE → END")
    return "end"


def build_graph(checkpointer=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=checkpointer)


graph = build_graph()
