"""Gera (ou completa) o índice Chroma localmente, fora do Streamlit.

Uso:
    python build_corpus.py

Requer GEMINI_API_KEY (ou OPENAI_API_KEY) no .env.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from src.pipeline.rag import build_rag_pipeline  # noqa: E402  (precisa vir depois do load_dotenv)


def main() -> None:
    pipeline = build_rag_pipeline()
    total = pipeline.collection.count()
    esperado = len(pipeline._build_chunks())
    print(f"\nÍndice em '{pipeline.persist_dir}': {total}/{esperado} chunks indexados.")
    if total < esperado:
        print("Indexação incompleta — rode o script de novo para completar o restante.")


if __name__ == "__main__":
    main()