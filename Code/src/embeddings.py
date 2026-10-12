"""Local Embeddings Pipeline.
Uses Sentence Transformers (e.g. all-MiniLM-L6-v2) for local text vectorization.
Includes lightweight fallback vectorizer to ensure zero-failure operation.
"""


import re
from typing import List, Optional
import numpy as np

import importlib.util

# Fast availability check — no actual import (avoids ~30s cold-start penalty from HuggingFace/transformers)
HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None


class LightweightFallbackEmbedding:
    """Deterministic token-frequency vectorizer ensuring RAG operation even if ST is uninstalled."""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_word(self, word: str) -> int:
        h = 0
        for ch in word:
            h = (h * 31 + ord(ch)) % self.dimension
        return h

    def embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if not words:
            return vec.tolist()
        for w in words:
            idx = self._hash_word(w)
            vec[idx] += 1.0
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._is_st_loaded = False
        self._fallback = LightweightFallbackEmbedding(dimension=384)
        self.dimension = 384
        self.device = "cpu"
        self.init_error = None

    def _lazy_load_model(self):
        """Lazy load SentenceTransformer model to optimize startup time."""
        if self._model is not None or not HAS_SENTENCE_TRANSFORMERS:
            return
        try:
            from sentence_transformers import SentenceTransformer  # deferred — avoids 30s import at startup
            self._model = SentenceTransformer(self.model_name)
            self._is_st_loaded = True
            if hasattr(self._model, "get_embedding_dimension"):
                self.dimension = self._model.get_embedding_dimension()
            else:
                self.dimension = self._model.get_sentence_embedding_dimension()
            self.device = str(self._model.device)
        except Exception as e:
            self.init_error = str(e)
            self._is_st_loaded = False
            self._model = None

    @property
    def is_using_sentence_transformers(self) -> bool:
        self._lazy_load_model()
        return self._is_st_loaded

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of document strings."""
        if not texts:
            return []
        self._lazy_load_model()
        if self._is_st_loaded and self._model is not None:
            try:
                embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
                return [e.tolist() for e in embeddings]
            except Exception:
                pass  # Fallback to deterministic vectorizer if inference fails
        
        return [self._fallback.embed_text(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Compute embedding for a single user query."""
        res = self.embed_documents([query])
        return res[0] if res else [0.0] * self.dimension
