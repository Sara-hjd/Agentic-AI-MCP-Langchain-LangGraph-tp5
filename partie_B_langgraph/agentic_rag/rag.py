"""Base de connaissances du chatbot : indexation du support de cours et recherche hybride.

PDF → chunks → (1) embeddings CLIP ViT-B-32 stockés dans Chroma (distance cosinus)
             → (2) index lexical BM25
Les deux classements sont fusionnés par Reciprocal Rank Fusion (RRF).
"""
import os
import re
from functools import lru_cache
from pathlib import Path

import fitz  # PyMuPDF
import open_clip
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

RACINE_TP5 = Path(__file__).resolve().parents[2]
PDF_PAR_DEFAUT = RACINE_TP5 / "Support Résumé Agentic AI et MCP - Pr. YOUSSFI.pdf"


class ClipEmbeddings(Embeddings):
    """Embeddings texte CLIP ViT-B-32 (laion2b_s34b_b79k), normalisés L2."""

    def __init__(self):
        self.model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval()

    @torch.no_grad()
    def embed_documents(self, texts):
        f = self.model.encode_text(self.tokenizer(texts))
        return (f / f.norm(dim=-1, keepdim=True)).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def _tokens(texte: str) -> list[str]:
    return re.findall(r"\w+", texte.lower())


class BaseConnaissances:
    """Index hybride (CLIP + BM25) sur un PDF."""

    def __init__(self, pdf: Path):
        pages = [Document(page_content=p.get_text(), metadata={"page": i + 1, "source": pdf.name})
                 for i, p in enumerate(fitz.open(pdf))]
        # CLIP n'encode que 77 tokens : petits chunks
        self.chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60).split_documents(pages)
        self.vector_store = Chroma.from_documents(
            self.chunks, ClipEmbeddings(), collection_name="cours_agentic_ai_langgraph",
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.bm25 = BM25Okapi([_tokens(c.page_content) for c in self.chunks])

    def recherche(self, query: str, k: int = 5, rrf_k: int = 60) -> list[Document]:
        scores, docs = {}, {}
        semantique = self.vector_store.similarity_search(query, k=10)
        scores_bm25 = self.bm25.get_scores(_tokens(query))
        lexical = [self.chunks[i] for i in sorted(range(len(self.chunks)), key=lambda i: -scores_bm25[i])[:10]]
        for classement in (semantique, lexical):
            for rang, d in enumerate(classement):
                cle = (d.metadata["page"], d.page_content)
                scores[cle] = scores.get(cle, 0) + 1 / (rrf_k + rang + 1)
                docs[cle] = d
        return [docs[c] for c in sorted(scores, key=scores.get, reverse=True)[:k]]


@lru_cache(maxsize=1)
def base_connaissances() -> BaseConnaissances:
    """Construit l'index une seule fois (chemin du PDF modifiable via RAG_PDF)."""
    return BaseConnaissances(Path(os.getenv("RAG_PDF", PDF_PAR_DEFAUT)))
