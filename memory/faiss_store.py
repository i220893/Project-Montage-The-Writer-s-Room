# ============================================================
# memory/faiss_store.py — FAISS Vector Store Wrapper
#
# Provides a singleton WritersRoomMemory object for semantic
# storage and retrieval of scripts, character profiles, and
# image references across sessions.
#
# Used directly inside the MCP server tools (search_memory,
# store_in_memory), but can also be imported by test scripts.
# ============================================================

import json
import logging
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBED_MODEL, FAISS_INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class WritersRoomMemory:
    """
    FAISS-backed persistent memory for the Writer's Room Pipeline.

    Stores:
      - Script loglines and scene summaries  (type: "script")
      - Character profiles and appearances   (type: "characters")
      - Image file paths and descriptions    (type: "image")

    The FAISS index is persisted to disk at FAISS_INDEX_PATH and
    automatically loaded on next startup.
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"trust_remote_code": True},
            show_progress=False,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self._store: FAISS | None = None
        self._load_existing()

    def _load_existing(self) -> None:
        """Load an existing FAISS index from disk if available."""
        if Path(FAISS_INDEX_PATH).exists():
            try:
                self._store = FAISS.load_local(
                    FAISS_INDEX_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("[Memory] Loaded existing FAISS index from %s", FAISS_INDEX_PATH)
            except Exception as exc:
                logger.warning("[Memory] Could not load FAISS index: %s", exc)
                self._store = None
        else:
            logger.info("[Memory] No existing FAISS index found. Will create on first store().")

    def add(self, text: str, metadata: dict | None = None) -> int:
        """
        Embed and store a text chunk (or multiple chunks if text is long).

        Args:
            text:     The content to embed.
            metadata: Optional dict (e.g. {"type": "script", "title": "..."})

        Returns:
            Number of chunks stored.
        """
        meta = metadata or {}
        docs = self.splitter.create_documents([text], metadatas=[meta])

        if self._store is None:
            self._store = FAISS.from_documents(docs, self.embeddings)
        else:
            self._store.add_documents(docs)

        self._save()
        logger.info("[Memory] Stored %d chunk(s). Metadata: %s", len(docs), meta)
        return len(docs)

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Semantic similarity search.

        Returns:
            List of up to k matching Document objects.
        """
        if self._store is None:
            logger.info("[Memory] Search called but store is empty.")
            return []
        results = self._store.similarity_search(query, k=k)
        logger.info("[Memory] Search for '%s' → %d result(s).", query, len(results))
        return results

    def search_with_scores(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Similarity search with distance scores."""
        if self._store is None:
            return []
        return self._store.similarity_search_with_score(query, k=k)

    def _save(self) -> None:
        """Persist the FAISS index to disk."""
        if self._store:
            Path(FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
            self._store.save_local(FAISS_INDEX_PATH)
            logger.debug("[Memory] FAISS index saved to %s.", FAISS_INDEX_PATH)

    @property
    def is_empty(self) -> bool:
        return self._store is None

    def clear(self) -> None:
        """Wipe the in-memory store (does NOT delete from disk)."""
        self._store = None
        logger.info("[Memory] In-memory FAISS store cleared.")


# ── Singleton instance used by MCP server tools ───────────────────────────────
memory_store = WritersRoomMemory()
