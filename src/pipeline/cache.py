"""Cache em 2 níveis: exact-match (SHA256) + semantic (cosine similarity)."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
from openai import OpenAI


class ExactCache:
    """Cache por hash SHA256 da query. Captura replays exatos."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        return self._store.get(self._key(query))

    def put(self, query: str, answer: str) -> None:
        self._store[self._key(query)] = answer

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store)}


class SemanticCache:
    """Cache por similaridade de embedding. Captura paráfrases."""

    def __init__(self, threshold: float = 0.93) -> None:
        self.threshold = threshold
        self._queries: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._answers: list[str] = []

        if "GEMINI_API_KEY" in os.environ:
            self._client = OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._embed_model = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
        else:
            self._client = OpenAI()
            self._embed_model = "text-embedding-3-small"

    def _embed(self, text: str) -> np.ndarray:
        r = self._client.embeddings.create(model=self._embed_model, input=text)
        return np.array(r.data[0].embedding)

    def get(self, query: str) -> str | None:
        """Retorna resposta cacheada se similar a alguma query anterior, ou None."""
        if not self._embeddings:
            return None

        e = self._embed(query)
        sims = [
            np.dot(e, em) / (np.linalg.norm(e) * np.linalg.norm(em))
            for em in self._embeddings
        ]
        idx = int(np.argmax(sims))

        if sims[idx] >= self.threshold:
            return self._answers[idx]
        return None

    def put(self, query: str, answer: str) -> None:
        embedding = self._embed(query)
        self._queries.append(query)
        self._embeddings.append(embedding)
        self._answers.append(answer)

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._queries), "threshold": self.threshold}
