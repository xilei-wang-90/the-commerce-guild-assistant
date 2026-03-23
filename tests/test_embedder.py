"""Tests for guild_assistant.rag_setup.embedder.Embedder."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from guild_assistant.rag_setup.embedder import Embedder


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _populate_dir(directory: Path, files: dict[str, str]) -> None:
    """Write *files* mapping ``{name: content}`` into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")


@pytest.fixture()
def mock_adapter():
    """Return a mock EmbeddingAdapter whose embed() returns a dummy vector."""
    adapter = MagicMock()
    adapter.embed.return_value = [0.1, 0.2, 0.3]
    return adapter


@pytest.fixture()
def mock_collection():
    """Return a mock ChromaDB collection with no pre-existing entries."""
    col = MagicMock()
    col.get.return_value = {"ids": []}
    return col


@pytest.fixture()
def mock_chroma(mock_collection, mocker):
    """Patch chromadb.PersistentClient. Returns the mock collection."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mocker.patch(
        "guild_assistant.rag_setup.embedder.chromadb.PersistentClient",
        return_value=mock_client,
    )
    return mock_collection


def _make_embedder(tmp_path, mock_chroma, mock_adapter) -> tuple[Embedder, Path, Path]:
    """Construct an Embedder with temp sources/silver directories."""
    sources_dir = tmp_path / "sources"
    silver_dir = tmp_path / "silver"
    db_path = tmp_path / "db"
    sources_dir.mkdir()
    silver_dir.mkdir()
    embedder = Embedder(
        sources_dir=sources_dir,
        silver_dir=silver_dir,
        db_path=db_path,
        embedding_adapter=mock_adapter,
        collection_name="test_collection",
    )
    return embedder, sources_dir, silver_dir


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestEmbedderInit:
    def test_creates_persistent_client_with_db_path(self, tmp_path, mock_adapter, mocker):
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = MagicMock()
        patched = mocker.patch(
            "guild_assistant.rag_setup.embedder.chromadb.PersistentClient",
            return_value=mock_client,
        )
        db_path = tmp_path / "mydb"
        Embedder(sources_dir=tmp_path, silver_dir=tmp_path, db_path=db_path,
                 embedding_adapter=mock_adapter, collection_name="test_collection")
        patched.assert_called_once_with(path=str(db_path))

    def test_gets_or_creates_collection_with_correct_name(self, tmp_path, mock_adapter, mocker):
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = MagicMock()
        mocker.patch(
            "guild_assistant.rag_setup.embedder.chromadb.PersistentClient",
            return_value=mock_client,
        )
        Embedder(sources_dir=tmp_path, silver_dir=tmp_path, db_path=tmp_path / "db",
                 embedding_adapter=mock_adapter, collection_name="my_collection")
        mock_client.get_or_create_collection.assert_called_once_with(name="my_collection", metadata={"hnsw:space": "cosine"})


# ---------------------------------------------------------------------------
# embed_file
# ---------------------------------------------------------------------------

class TestEmbedFile:
    def test_calls_adapter_embed_with_content(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        content = "Yakmel summary text."
        _populate_dir(summaries_dir, {"yakmel.md": content})

        embedder.embed_file(summaries_dir / "yakmel.md")

        mock_adapter.embed.assert_called_once_with(content)

    def test_upserts_document_with_correct_id(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {"yakmel.md": "Yakmel summary text."})

        embedder.embed_file(summaries_dir / "yakmel.md")

        kwargs = mock_chroma.upsert.call_args.kwargs
        assert kwargs["ids"] == ["yakmel"]

    def test_upserts_adapter_vector(self, tmp_path, mock_chroma, mock_adapter):
        mock_adapter.embed.return_value = [0.5, 0.6, 0.7]
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {"yakmel.md": "text"})

        embedder.embed_file(summaries_dir / "yakmel.md")

        kwargs = mock_chroma.upsert.call_args.kwargs
        assert kwargs["embeddings"] == [[0.5, 0.6, 0.7]]

    def test_upserts_document_text(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        content = "Detailed summary about yakmel."
        _populate_dir(summaries_dir, {"yakmel.md": content})

        embedder.embed_file(summaries_dir / "yakmel.md")

        kwargs = mock_chroma.upsert.call_args.kwargs
        assert kwargs["documents"] == [content]

    def test_metadata_contains_filename(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {"yakmel.md": "text"})

        embedder.embed_file(summaries_dir / "yakmel.md")

        metadata = mock_chroma.upsert.call_args.kwargs["metadatas"][0]
        assert metadata["filename"] == "yakmel.md"

    def test_metadata_contains_silver_path(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, silver_dir = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {"yakmel.md": "text"})

        embedder.embed_file(summaries_dir / "yakmel.md")

        metadata = mock_chroma.upsert.call_args.kwargs["metadatas"][0]
        assert metadata["silver_path"] == str(silver_dir / "yakmel.md")

    def test_metadata_contains_summary_path(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {"yakmel.md": "text"})

        embedder.embed_file(summaries_dir / "yakmel.md")

        metadata = mock_chroma.upsert.call_args.kwargs["metadatas"][0]
        assert metadata["summary_path"] == str(summaries_dir / "yakmel.md")

    def test_returns_summary_text(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        content = "Returned summary text."
        _populate_dir(summaries_dir, {"page.md": content})

        result = embedder.embed_file(summaries_dir / "page.md")

        assert result == content


# ---------------------------------------------------------------------------
# embed_all
# ---------------------------------------------------------------------------

class TestEmbedAll:
    def test_embeds_all_md_files(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {"a.md": "alpha", "b.md": "beta"})

        results = embedder.embed_all()

        assert set(results.keys()) == {"a.md", "b.md"}
        assert mock_chroma.upsert.call_count == 2

    def test_skips_already_embedded(self, tmp_path, mock_chroma, mock_adapter):
        mock_chroma.get.side_effect = lambda ids: (
            {"ids": ids} if ids == ["existing"] else {"ids": []}
        )
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {"existing.md": "old", "new.md": "new text"})

        results = embedder.embed_all()

        assert list(results.keys()) == ["new.md"]
        assert mock_chroma.upsert.call_count == 1

    def test_force_reupserts_existing(self, tmp_path, mock_chroma, mock_adapter):
        mock_chroma.get.return_value = {"ids": ["existing"]}
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {"existing.md": "content"})

        results = embedder.embed_all(force=True)

        assert "existing.md" in results
        mock_chroma.upsert.assert_called_once()

    def test_empty_directory_returns_empty(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)

        results = embedder.embed_all()

        assert results == {}
        mock_chroma.upsert.assert_not_called()

    def test_returns_dict_of_filename_to_text(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {"page.md": "summary content"})

        results = embedder.embed_all()

        assert results == {"page.md": "summary content"}

    def test_continues_on_single_file_failure(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {"aaa.md": "good", "bbb.md": "bad", "ccc.md": "good"})

        mock_chroma.upsert.side_effect = [
            None,
            RuntimeError("chroma exploded"),
            None,
        ]

        results = embedder.embed_all()

        assert set(results.keys()) == {"aaa.md", "ccc.md"}

    def test_force_skips_collection_get_check(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {"page.md": "text"})

        embedder.embed_all(force=True)

        mock_chroma.get.assert_not_called()

    def test_max_records_limits_embedded(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {
            "a.md": "a", "b.md": "b", "c.md": "c", "d.md": "d", "e.md": "e",
        })

        results = embedder.embed_all(max_records=3)

        assert len(results) == 3
        assert mock_chroma.upsert.call_count == 3

    def test_max_records_zero_embeds_all(self, tmp_path, mock_chroma, mock_adapter):
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {"a.md": "a", "b.md": "b", "c.md": "c"})

        results = embedder.embed_all(max_records=0)

        assert len(results) == 3

    def test_max_records_skipped_dont_count(self, tmp_path, mock_chroma, mock_adapter):
        """Already-embedded (skipped) files don't count toward max_records."""
        mock_chroma.get.side_effect = lambda ids: (
            {"ids": ids} if ids == ["done"] else {"ids": []}
        )
        embedder, summaries_dir, _ = _make_embedder(tmp_path, mock_chroma, mock_adapter)
        _populate_dir(summaries_dir, {
            "done.md": "old", "a.md": "a", "b.md": "b", "c.md": "c",
        })

        results = embedder.embed_all(max_records=2)

        assert len(results) == 2
        assert "done.md" not in results
