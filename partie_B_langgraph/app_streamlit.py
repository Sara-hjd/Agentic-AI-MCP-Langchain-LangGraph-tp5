"""Interface web Streamlit du chatbot Agentic RAG (LangGraph).

Lancement (depuis partie_B_langgraph/) :
    uv run streamlit run app_streamlit.py
"""
import uuid

import streamlit as st
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agentic_rag.graph import build_graph
from agentic_rag.rag import base_connaissances

st.set_page_config(page_title="Chatbot Agentic RAG", page_icon="🤖", layout="centered")


@st.cache_resource(show_spinner="Indexation du support de cours (CLIP + BM25)…")
def charger_graphe():
    base_connaissances()  # construit l'index une seule fois
    return build_graph(checkpointer=InMemorySaver())  # mémoire de conversation par thread_id


graph = charger_graphe()

# ----------------------------------------------------------------------------- état de la session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.historique = []  # [(role, contenu, details_outils)]

with st.sidebar:
    st.header("⚙️ Session")
    st.caption(f"thread_id : `{st.session_state.thread_id[:8]}`")
    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.historique = []
        st.rerun()
    st.divider()
    st.markdown(
        "**Architecture**\n\n"
        "`START → agent ⇄ tools → END`\n\n"
        "- **agent** : qwen3-vl:8b (Ollama)\n"
        "- **tools** : `retriever_tool` (CLIP + BM25, Chroma)\n"
        "- **mémoire** : `InMemorySaver` par `thread_id`"
    )

st.title("🤖 Chatbot Agentic RAG")
st.caption("Questions sur le support de cours « Agentic AI et MCP » — LangGraph + Ollama")


def afficher_details(details):
    if details:
        with st.expander(f"🔎 {len(details)} recherche(s) dans le cours"):
            for requete, passages in details:
                st.markdown(f"**Requête :** `{requete}`")
                st.text(passages[:1500])


for role, contenu, details in st.session_state.historique:
    with st.chat_message(role):
        st.markdown(contenu)
        afficher_details(details)

# ----------------------------------------------------------------------------- nouveau message
if question := st.chat_input("Posez votre question sur le cours…"):
    st.session_state.historique.append(("user", question, []))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        details, requetes, reponse = [], {}, ""
        with st.status("L'agent réfléchit…", expanded=False) as statut:
            # stream_mode="updates" : une mise à jour par nœud exécuté (agent / tools)
            for etape in graph.stream({"messages": [HumanMessage(question)]}, config, stream_mode="updates"):
                for noeud, maj in etape.items():
                    for m in maj["messages"]:
                        if isinstance(m, AIMessage) and m.tool_calls:
                            for tc in m.tool_calls:
                                requetes[tc["id"]] = tc["args"].get("query", "")
                                statut.write(f"🛠 Recherche : `{requetes[tc['id']]}`")
                        elif isinstance(m, ToolMessage):
                            details.append((requetes.get(m.tool_call_id, ""), m.content))
                            statut.write("👁 Passages récupérés")
                        elif isinstance(m, AIMessage):
                            reponse = m.content
            statut.update(label="Réponse prête", state="complete")
        st.markdown(reponse)
        afficher_details(details)
    st.session_state.historique.append(("assistant", reponse, details))
