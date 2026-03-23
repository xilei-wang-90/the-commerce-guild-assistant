"""Embed ``.md`` files into a persistent ChromaDB vector store.

Reads ``.md`` files from a sources directory, embeds them via an
``EmbeddingAdapter``, and upserts each document into a ChromaDB collection.
The collection is persisted to disk so it survives across runs.

Each ChromaDB entry carries metadata pointing back to the original full-text
file in the silver tier, enabling downstream retrieval to fetch the full
article after a similarity match on the source embedding.
"""

import logging
from pathlib import Path

import chromadb

from guild_assistant.utils.model_adapter import EmbeddingAdapter

logger = logging.getLogger(__name__)


class Embedder:
    """Read ``.md`` files from *sources_dir*, embed them, and store in ChromaDB.

    *sources_dir*      – directory containing the ``.md`` files to embed
                         (e.g. LLM-generated summaries or raw silver files).
    *silver_dir*       – directory containing the full-text silver-tier files
                         (same filenames as the sources).
    *db_path*          – path where the ChromaDB database is persisted.
    *embedding_adapter* – ``EmbeddingAdapter`` used to compute vectors.
    *collection_name*  – ChromaDB collection to use (created if absent).
    """

    def __init__(
        self,
        sources_dir: str | Path,
        silver_dir: str | Path,
        db_path: str | Path,
        embedding_adapter: EmbeddingAdapter,
        collection_name: str,
    ) -> None:
        self._sources_dir = Path(sources_dir)
        self._silver_dir = Path(silver_dir)
        self._adapter = embedding_adapter

        client = chromadb.PersistentClient(path=str(Path(db_path)))
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_file(self, file_path: Path) -> str:
        """Embed a single source file and upsert it into the collection.

        The document ID is the filename stem (without ``.md``).  Metadata
        includes the filename, the path to the embedded source file (stored
        under ``summary_path`` for historical reasons — may be a summary or a
        reverse-hyde question file), and the path to the corresponding
        full-text silver file (``silver_path``).

        Returns the source text that was embedded.
        """
        doc_id = file_path.stem
        content = file_path.read_text(encoding="utf-8")
        silver_path = str(self._silver_dir / file_path.name)
        vector = self._adapter.embed(content)

        self._collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            documents=[content],
            metadatas=[{
                "filename": file_path.name,
                "summary_path": str(file_path),
                "silver_path": silver_path,
            }],
        )
        logger.info("Embedded %s", file_path.name)
        return content

    def embed_all(self, force: bool = False, max_records: int = 10) -> dict[str, str]:
        """Embed ``.md`` files in *sources_dir*.

        Files whose ID already exists in the collection are skipped unless
        *force* is ``True``, in which case they are re-embedded via upsert.
        Skipped files do not count toward *max_records*.

        *max_records* caps the number of files actually embedded.  Set to
        ``0`` to embed all eligible files.

        Returns a mapping of ``{filename: text}`` for every file that was
        (re-)embedded in this run.
        """
        results: dict[str, str] = {}
        md_files = sorted(self._sources_dir.glob("*.md"))

        if not md_files:
            logger.warning("No .md files found in %s", self._sources_dir)
            return results

        logger.info("Found %d .md files to embed", len(md_files))

        for file_path in md_files:
            if max_records and len(results) >= max_records:
                break
            doc_id = file_path.stem
            if not force:
                existing = self._collection.get(ids=[doc_id])
                if existing["ids"]:
                    logger.info("Skipping %s (already embedded)", file_path.name)
                    continue
            try:
                content = self.embed_file(file_path)
                results[file_path.name] = content
            except Exception:
                logger.exception("Failed to embed %s", file_path.name)

        logger.info("Finished: %d embedded, %d total", len(results), len(md_files))
        return results
