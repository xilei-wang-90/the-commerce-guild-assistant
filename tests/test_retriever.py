"""Tests for guild_assistant.rag.retriever."""

from unittest.mock import MagicMock, patch

import pytest

from guild_assistant.rag.retriever import Retriever, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_adapter():
    adapter = MagicMock()
    adapter.embed.return_value = [0.1, 0.2, 0.3]
    return adapter


@pytest.fixture()
def mock_collection():
    col = MagicMock()
    col.query.return_value = {
        "ids": [["doc1", "doc2"]],
        "documents": [["Summary one.", "Summary two."]],
        "metadatas": [[
            {"silver_path": "/data/silver/doc1.md"},
            {"silver_path": "/data/silver/doc2.md"},
        ]],
        "distances": [[0.1, 0.3]],
    }
    return col


@pytest.fixture()
def mock_chroma(mock_collection, mocker):
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mocker.patch(
        "guild_assistant.rag.retriever.chromadb.PersistentClient",
        return_value=mock_client,
    )
    return mock_collection


def _make_retriever(mock_chroma, mock_adapter, n_results=3):
    return Retriever(
        db_path="/tmp/db",
        collection_name="test_col",
        embedding_adapter=mock_adapter,
        n_results=n_results,
    )


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_returns_retrieval_results(self, mock_chroma, mock_adapter):
        retriever = _make_retriever(mock_chroma, mock_adapter)
        results = retriever.retrieve("What is a Yakmel?")

        assert len(results) == 2
        assert isinstance(results[0], RetrievalResult)

    def test_result_fields(self, mock_chroma, mock_adapter):
        retriever = _make_retriever(mock_chroma, mock_adapter)
        results = retriever.retrieve("query")

        assert results[0].doc_id == "doc1"
        assert results[0].summary == "Summary one."
        assert results[0].silver_path == "/data/silver/doc1.md"
        assert results[0].distance == 0.1

    def test_embeds_query_with_adapter(self, mock_chroma, mock_adapter):
        retriever = _make_retriever(mock_chroma, mock_adapter)
        retriever.retrieve("my question")

        mock_adapter.embed.assert_called_once_with("my question")

    def test_passes_n_results_to_collection(self, mock_chroma, mock_adapter):
        retriever = _make_retriever(mock_chroma, mock_adapter, n_results=5)
        retriever.retrieve("query")

        mock_chroma.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=5,
        )

    def test_empty_results(self, mock_chroma, mock_adapter):
        mock_chroma.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        retriever = _make_retriever(mock_chroma, mock_adapter)
        results = retriever.retrieve("obscure query")

        assert results == []

    def test_custom_n_results(self, mock_chroma, mock_adapter):
        retriever = _make_retriever(mock_chroma, mock_adapter, n_results=1)
        assert retriever._n_results == 1
