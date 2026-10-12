"""Vector Database and Indexing Module.
Manages persistent chunk storage and similarity search using ChromaDB.
Includes duplicate detection, metadata filtering, and safe document deletion.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from src.chunking import TextChunk
from src.embeddings import EmbeddingManager

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class VectorStoreManager:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_manager: Optional[EmbeddingManager] = None,
        collection_name: str = "autosar_hld_chunks",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        # In-memory / lightweight fallback cache if ChromaDB unavailable
        self._fallback_store: Dict[str, Dict[str, Any]] = {}
        self._init_store()

    def _init_store(self):
        """Initialize ChromaDB PersistentClient or local fallback."""
        if HAS_CHROMADB:
            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory)
                )
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                return
            except Exception:
                self._client = None
                self._collection = None

    def is_document_indexed(self, doc_hash: str) -> bool:
        """Check if a document with this exact SHA-256 hash has already been indexed."""
        if self._collection is not None:
            try:
                results = self._collection.get(
                    where={"doc_hash": doc_hash},
                    limit=1,
                )
                return len(results.get("ids", [])) > 0
            except Exception:
                pass
        
        # Check fallback
        for item in self._fallback_store.values():
            if item.get("metadata", {}).get("doc_hash") == doc_hash:
                return True
        return False

    def index_chunks(self, chunks: List[TextChunk]) -> int:
        """Index a list of chunks, preventing duplicate indexing."""
        if not chunks:
            return 0

        # Check if already indexed
        doc_hash = chunks[0].doc_hash
        if self.is_document_indexed(doc_hash):
            return 0  # Skip duplicate

        texts = [c.text for c in chunks]
        embeddings = self.embedding_manager.embed_documents(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]

        if self._collection is not None:
            try:
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                return len(chunks)
            except Exception as e:
                # Fallback to in-memory store if ChromaDB writes fail
                pass

        # Fallback indexing
        for i, cid in enumerate(ids):
            self._fallback_store[cid] = {
                "id": cid,
                "text": texts[i],
                "embedding": embeddings[i],
                "metadata": metadatas[i],
            }
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 4,
        doc_id_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks with similarity scores and metadata."""
        query_embedding = self.embedding_manager.embed_query(query)
        retrieved: List[Dict[str, Any]] = []

        if self._collection is not None:
            try:
                where_clause = {"doc_id": doc_id_filter} if doc_id_filter else None
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_clause,
                    include=["documents", "metadatas", "distances"],
                )
                if results and results.get("ids") and len(results["ids"][0]) > 0:
                    for idx in range(len(results["ids"][0])):
                        dist = results["distances"][0][idx] if "distances" in results else 0.0
                        # Convert cosine distance to similarity score
                        sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                        retrieved.append({
                            "chunk_id": results["ids"][0][idx],
                            "text": results["documents"][0][idx],
                            "metadata": results["metadatas"][0][idx],
                            "similarity": float(sim),
                            "distance": float(dist),
                        })
                    return retrieved
            except Exception:
                pass

        # Fallback vector search via dot-product cosine similarity
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        scored = []
        for item in self._fallback_store.values():
            if doc_id_filter and item["metadata"].get("doc_id") != doc_id_filter:
                continue
            d_vec = np.array(item["embedding"], dtype=np.float32)
            d_norm = np.linalg.norm(d_vec)
            if q_norm > 0 and d_norm > 0:
                score = float(np.dot(q_vec, d_vec) / (q_norm * d_norm))
            else:
                score = 0.0
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        for score, item in scored[:top_k]:
            retrieved.append({
                "chunk_id": item["id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "similarity": max(0.0, min(1.0, (score + 1.0) / 2.0)),
                "distance": max(0.0, 1.0 - score),
            })
        return retrieved

    def delete_document(self, doc_id: str) -> bool:
        """Safely remove all chunks belonging to a document without corrupting index."""
        deleted = False
        if self._collection is not None:
            try:
                self._collection.delete(where={"doc_id": doc_id})
                deleted = True
            except Exception:
                pass

        keys_to_delete = [
            k for k, v in self._fallback_store.items()
            if v.get("metadata", {}).get("doc_id") == doc_id
        ]
        for k in keys_to_delete:
            del self._fallback_store[k]
            deleted = True
        return deleted

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """Return list of distinct indexed documents and their chunk counts."""
        doc_map = {}
        if self._collection is not None:
            try:
                all_data = self._collection.get(include=["metadatas"])
                for meta in all_data.get("metadatas", []):
                    did = meta.get("doc_id")
                    if did:
                        if did not in doc_map:
                            doc_map[did] = {
                                "doc_id": did,
                                "filename": meta.get("filename", "unknown"),
                                "doc_hash": meta.get("doc_hash", ""),
                                "chunk_count": 0,
                            }
                        doc_map[did]["chunk_count"] += 1
                return list(doc_map.values())
            except Exception:
                pass

        for item in self._fallback_store.values():
            meta = item.get("metadata", {})
            did = meta.get("doc_id")
            if did:
                if did not in doc_map:
                    doc_map[did] = {
                        "doc_id": did,
                        "filename": meta.get("filename", "unknown"),
                        "doc_hash": meta.get("doc_hash", ""),
                        "chunk_count": 0,
                    }
                doc_map[did]["chunk_count"] += 1
        return list(doc_map.values())

    def get_total_chunk_count(self) -> int:
        """Return total number of chunks currently stored in index."""
        if self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                pass
        return len(self._fallback_store)
