"""RAG pipeline — chunk, embed, index, retrieve, generate."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader


def _make_client() -> tuple[OpenAI, str]:
    if "GEMINI_API_KEY" in os.environ:
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        embed_api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif "OPENAI_API_KEY" in os.environ:
        client = OpenAI()
        embed_api_base = None
    else:
        raise RuntimeError("Configure GEMINI_API_KEY ou OPENAI_API_KEY no .env")
    return client, embed_api_base


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.client, embed_api_base = _make_client()
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        self.embed_model = embed_model or os.environ.get("EMBED_MODEL", "gemini-embedding-001")

        embed_kwargs: dict[str, Any] = {
            "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            "model_name": self.embed_model,
        }
        if embed_api_base:
            embed_kwargs["api_base"] = embed_api_base
        self.embed_fn = OpenAIEmbeddingFunction(**embed_kwargs)

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    def _build_chunks(self) -> list[dict]:
        """Lê PDFs e TXTs de `corpus_dir`, faz chunking. Não chama a API de embedding."""
        docs: list[dict] = []
    
        for pdf_path in sorted(self.corpus_dir.glob("*.pdf")):
            reader = PdfReader(pdf_path)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({"text": text, "source": pdf_path.name, "page": page_idx + 1})
    
        for txt_path in sorted(self.corpus_dir.glob("*.txt")):
            full_text = txt_path.read_text(encoding="utf-8")
            blocks = [b.strip() for b in full_text.split("---") if b.strip()]
            for i, block in enumerate(blocks):
                docs.append({"text": block, "source": txt_path.name, "page": i + 1})
    
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
        chunks: list[dict] = []
        for doc in docs:
            for i, chunk in enumerate(splitter.split_text(doc["text"])):
                chunks.append({
                    "id": f"{doc['source']}-p{doc['page']}-c{i}",
                    "text": chunk,
                    "source": doc["source"],
                    "page": doc["page"],
                })
        return chunks

    def ingest_and_index(self) -> int:
        """Faz chunking e indexa em Chroma."""
        chunks = self._build_chunks()
        print(f"\n-- Iniciando indexação: {len(chunks)} chunks --\n")
    
        BATCH_SIZE = 10
        DELAY_SECONDS = 4.5
    
        for i in range(0, len(chunks), BATCH_SIZE):
            lote = chunks[i : i + BATCH_SIZE]
            for tentativa in range(3):
                try:
                    self.collection.add(
                        ids=[c["id"] for c in lote],
                        documents=[c["text"] for c in lote],
                        metadatas=[{"source": c["source"], "page": c["page"]} for c in lote],
                    )
                    break
                except Exception as e:
                    print(f"Erro no batch {i}: {e}. Retry em {DELAY_SECONDS * 2}s...")
                    time.sleep(DELAY_SECONDS * 2)
            time.sleep(DELAY_SECONDS)
    
        total_indexado = self.collection.count()
        if total_indexado < len(chunks):
            print(f"AVISO: indexação incompleta ({total_indexado}/{len(chunks)} chunks).")
    
        print("\n-- Indexação concluída --\n")
        return total_indexado

    def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
        """Cria pipeline e indexa corpus se ainda não indexado (ou incompleto)."""
        pipeline = RAGPipeline(corpus_dir=corpus_dir)
        total_esperado = len(pipeline._build_chunks())
        if pipeline.collection.count() < total_esperado:
            pipeline.ingest_and_index()
        return pipeline

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Busca top-k chunks similares à query."""
        result = self.collection.query(query_texts=[query], n_results=k)
        return [
            {
                "text": result["documents"][0][i],
                "source": result["metadatas"][0][i]["source"],
                "distance": result["distances"][0][i],
            }
            for i in range(len(result["documents"][0]))
        ]

    def answer(self, question: str, k: int = 5, model: str | None = None) -> dict:
        """Retrieve + augment + generate. Retorna {answer, sources}.

        `model` permite ao caller sobrescrever o llm_model default
        sem mutar o estado do pipeline.
        """
        hits = self.retrieve(question, k=k)
        context = "\n\n---\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        response = self.client.chat.completions.create(
            model=model or self.llm_model,
            messages=[
                {"role": "user", "content": PROMPT_TEMPLATE.format(context=context, question=question)}
            ],
            temperature=0.0,
        )
        return {
            "answer": response.choices[0].message.content.strip(),
            "sources": [h["source"] for h in hits],
        }


PROMPT_TEMPLATE = """Você é o Holocron, um assistente especializado nos 6 filmes da saga Star Wars (Episódios I–VI).
Responda APENAS com base no contexto abaixo.
Se a informação não estiver no contexto, diga "Não encontrado no corpus do Holocron".
Sempre cite a fonte usando o formato [arquivo].

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Cria pipeline e indexa corpus se ainda não indexado."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() < 50:
        pipeline.ingest_and_index()
    return pipeline
