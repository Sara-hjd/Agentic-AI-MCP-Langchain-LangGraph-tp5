"""Agent RAG multimodal sur les manuels du SAV (utilisé derrière le tool MCP `rag_multimodal`).

Indexation (une seule fois) :
    manuels PDF ──► passages (texte)  ──► CLIP texte  ─┐
               └──► figures (images) ──► CLIP image  ─┴─► ChromaDB (cosinus)   + BM25 sur les passages

Agent RAG (LangGraph) :   START → recherche → generation → END
- recherche  : passages (CLIP + BM25, fusion RRF) et figures (requête texte, et photo du client si fournie :
               similarité CLIP image ↔ image) ;
- generation : qwen3-vl:8b reçoit les passages, la figure retenue et la photo, puis rédige une réponse sourcée.
"""
import base64
import io
import re
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import chromadb
import fitz  # PyMuPDF
import open_clip
import torch
from langchain.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from PIL import Image
from rank_bm25 import BM25Okapi

ICI = Path(__file__).resolve().parent
DOSSIER_MANUELS = ICI / "donnees" / "manuels"
DOSSIER_IMAGES = ICI / "donnees" / "images"  # photos envoyées par les clients (référence = nom du fichier)


# ----------------------------------------------------------------------------- CLIP
class Clip:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval()

    @torch.no_grad()
    def textes(self, textes):
        f = self.model.encode_text(self.tokenizer(textes))
        return (f / f.norm(dim=-1, keepdim=True)).tolist()

    @torch.no_grad()
    def images(self, images):
        f = self.model.encode_image(torch.stack([self.preprocess(i.convert("RGB")) for i in images]))
        return (f / f.norm(dim=-1, keepdim=True)).tolist()


def _tokens(texte):
    return re.findall(r"\w+", texte.lower())


# ----------------------------------------------------------------------------- index
class IndexManuels:
    def __init__(self, dossier: Path = DOSSIER_MANUELS):
        self.clip = Clip()
        self.passages, self.figures = [], []  # dicts : {id, texte|image, source, page, ...}
        for pdf in sorted(dossier.glob("*.pdf")):
            self._extraire(pdf)
        client = chromadb.EphemeralClient()
        self.col_passages = client.get_or_create_collection("passages", metadata={"hnsw:space": "cosine"})
        self.col_figures = client.get_or_create_collection("figures", metadata={"hnsw:space": "cosine"})
        self.col_passages.add(ids=[p["id"] for p in self.passages],
                              embeddings=self.clip.textes([p["texte"] for p in self.passages]))
        self.col_figures.add(ids=[f["id"] for f in self.figures],
                             embeddings=self.clip.images([f["image"] for f in self.figures]))
        self.bm25 = BM25Okapi([_tokens(p["texte"]) for p in self.passages])

    def _extraire(self, pdf: Path):
        doc = fitz.open(pdf)
        for num, page in enumerate(doc, start=1):
            # passages = paragraphes de la page (blocs de texte)
            for k, bloc in enumerate(b for b in page.get_text("blocks") if b[4].strip()):
                texte = " ".join(bloc[4].split())
                self.passages.append({"id": f"{pdf.stem}-p{num}-b{k}", "texte": texte, "source": pdf.name, "page": num})
            # figures = images de la page + légende (« Figure N - ... ») située juste en dessous
            blocs = page.get_text("blocks")
            for info in page.get_image_info(xrefs=True):
                x0, y0, x1, y1 = info["bbox"]
                legende = min((b for b in blocs if b[1] >= y1 - 2 and b[4].strip()), key=lambda b: b[1], default=None)
                legende = " ".join(legende[4].split()) if legende else ""
                numero = re.match(r"Figure (\d+)", legende)
                pix = fitz.Pixmap(doc, info["xref"])
                self.figures.append({
                    "id": f"{pdf.stem}-fig{numero.group(1) if numero else info['xref']}",
                    "figure": f"Figure {numero.group(1)}" if numero else "Figure",
                    "legende": legende, "source": pdf.name, "page": num,
                    "image": Image.open(io.BytesIO(pix.tobytes("png"))),
                })

    def rechercher_passages(self, requete, k=6, rrf_k=60):
        scores = {}
        semantique = self.col_passages.query(query_embeddings=self.clip.textes([requete]), n_results=10)["ids"][0]
        s_bm25 = self.bm25.get_scores(_tokens(requete))
        lexical = [self.passages[i]["id"] for i in sorted(range(len(self.passages)), key=lambda i: -s_bm25[i])[:10]]
        for classement in (semantique, lexical):
            for rang, pid in enumerate(classement):
                scores[pid] = scores.get(pid, 0) + 1 / (rrf_k + rang + 1)
        par_id = {p["id"]: p for p in self.passages}
        return [par_id[pid] for pid in sorted(scores, key=scores.get, reverse=True)[:k]]

    def rechercher_figures(self, requete=None, image=None, k=1):
        """Avec une photo : similarité image ↔ image ; sinon : similarité texte ↔ image."""
        emb = self.clip.images([image]) if image is not None else self.clip.textes([requete])
        res = self.col_figures.query(query_embeddings=emb, n_results=k, include=["distances"])
        par_id = {f["id"]: f for f in self.figures}
        return [dict(par_id[i], similarite=round(1 - d, 3)) for i, d in zip(res["ids"][0], res["distances"][0])]


@lru_cache(maxsize=1)
def index_manuels() -> IndexManuels:
    return IndexManuels()


def charger_image_client(image_ref: str) -> Image.Image | None:
    """Résout une référence d'image client (ex. « img_px200_01 ») vers le fichier correspondant."""
    if not image_ref:
        return None
    fichiers = sorted(DOSSIER_IMAGES.glob(f"{Path(image_ref).stem}.*"))
    return Image.open(fichiers[0]) if fichiers else None


def _data_url(image: Image.Image) -> str:
    tampon = io.BytesIO()
    image.convert("RGB").save(tampon, format="PNG")
    return "data:image/png;base64," + base64.b64encode(tampon.getvalue()).decode()


# ----------------------------------------------------------------------------- agent RAG
class EtatRAG(TypedDict, total=False):
    question: str
    image_ref: str
    passages: list
    figures: list
    reponse: str


llm = ChatOllama(model="qwen3-vl:8b", temperature=0, reasoning=False, num_ctx=16384)

CONSIGNES = """Tu es l'agent documentaire du service après-vente. Réponds à la question du client UNIQUEMENT
à partir des passages et des figures du manuel fournis. Si une photo du client est fournie, compare-la à la
figure du manuel pour identifier le pictogramme. Termine par une ligne « Sources : » listant le manuel,
les pages et les figures utilisées (ex. « manuel_PX-200.pdf, page 3, Figure 1 »). Réponds en français."""


def noeud_recherche(etat: EtatRAG) -> EtatRAG:
    index = index_manuels()
    photo = charger_image_client(etat.get("image_ref", ""))
    figures = index.rechercher_figures(image=photo) if photo is not None else []
    # la légende de la figure identifiée enrichit la requête textuelle (ex. « bourrage papier E-02 »)
    requete = etat["question"] + (" " + figures[0]["legende"] if figures else "")
    passages = index.rechercher_passages(requete)
    if not figures:
        figures = index.rechercher_figures(requete=etat["question"])
    return {"passages": passages, "figures": figures}


def noeud_generation(etat: EtatRAG) -> EtatRAG:
    contexte = "\n".join(f"- [{p['source']}, page {p['page']}] {p['texte']}" for p in etat["passages"])
    contenu = [{"type": "text", "text": f"Question du client : {etat['question']}\n\nPassages du manuel :\n{contexte}"}]
    for f in etat["figures"]:
        contenu.append({"type": "text", "text": f"\nFigure du manuel ({f['source']}, page {f['page']}, "
                                                f"{f['figure']}) — légende : {f['legende']}"})
        contenu.append({"type": "image_url", "image_url": _data_url(f["image"])})
    photo = charger_image_client(etat.get("image_ref", ""))
    if photo is not None:
        contenu.append({"type": "text", "text": f"\nPhoto envoyée par le client ({etat['image_ref']}) :"})
        contenu.append({"type": "image_url", "image_url": _data_url(photo)})
    reponse = llm.invoke([SystemMessage(CONSIGNES), HumanMessage(content=contenu)])
    return {"reponse": reponse.content}


def _construire_agent_rag():
    g = StateGraph(EtatRAG)
    g.add_node("recherche", noeud_recherche)
    g.add_node("generation", noeud_generation)
    g.add_edge(START, "recherche")
    g.add_edge("recherche", "generation")
    g.add_edge("generation", END)
    return g.compile()


agent_rag = _construire_agent_rag()


def repondre(question: str, image_ref: str = "") -> str:
    """Point d'entrée du tool MCP : exécute l'agent RAG et renvoie la réponse sourcée."""
    etat = agent_rag.invoke({"question": question, "image_ref": image_ref})
    figures = ", ".join(f"{f['figure']} (page {f['page']}, similarité {f['similarite']})" for f in etat["figures"])
    return f"{etat['reponse']}\n\n[figures consultées : {figures}]"
