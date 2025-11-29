# This test calls chat() directly, but mocks embeddings + Pinecone.
# tests/unit/test_rag_pipeline.py
from unittest.mock import patch, MagicMock
from api.services.rag_pipeline import chat
import pandas as pd


def test_chat_function_builds_payload_correctly():
    print("Running test_chat_function_builds_payload_correctly() ...")

    # mock embedding output
    dummy_vec = [0.1, 0.2, 0.3]

    # mock Pinecone results
    dummy_results = {
        "documents": [["chunk1 text", "chunk2 text"]],
        "metadatas": [[{"bug": "mosquito"}]],
    }

    with patch("api.services.rag_pipeline.generate_query_embedding", return_value=dummy_vec):
        with patch("api.services.rag_pipeline.query_by_vector", return_value=dummy_results):

            payload = chat(
                symptoms="itchy red bumps",
                conf=0.9,
                bug_class="mosquito",
            )

    assert payload["bug_class"] == "mosquito"
    assert "itchy red bumps" in payload["prompt"]
    assert "chunk1 text" in payload["context"]
    assert "prompt" in payload
    assert "context" in payload
    assert "question" in payload

    print("Test passed.")


def test_agent_function_builds_payload_correctly():
    from api.services.rag_pipeline import agent

    dummy_vec = [0.1, 0.2, 0.3]
    dummy_results = {
        "documents": [["chunk1 text", "chunk2 text"]],
        "metadatas": [[{"bug": "mosquito"}]],
    }
    with patch("api.services.rag_pipeline.generate_query_embedding", return_value=dummy_vec):
        with patch("api.services.rag_pipeline.query_by_vector", return_value=dummy_results):
            payload = agent(question="How to prevent?", where={"type": {"$eq": "mosquito bites"}})
    assert "prompt" in payload
    assert "context" in payload
    assert "question" in payload


def test_generate_query_embedding():
    from api.services.rag_pipeline import generate_query_embedding

    with patch("api.services.rag_pipeline.llm_client") as mock_llm:
        mock_llm.models.embed_content.return_value.embeddings = [type("E", (), {"values": [0.1, 0.2, 0.3]})()]
        result = generate_query_embedding("test query")
    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3]


def test_generate_text_embeddings():
    from api.services.rag_pipeline import generate_text_embeddings

    with patch("api.services.rag_pipeline.llm_client") as mock_llm:
        mock_embed = type("E", (), {"values": [0.1, 0.2]})()
        mock_llm.models.embed_content.return_value.embeddings = [mock_embed, mock_embed]
        result = generate_text_embeddings(["chunk1", "chunk2"], batch_size=1)
    assert isinstance(result, list)
    assert result == [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]


def test_load_text_embeddings():
    from api.services.rag_pipeline import load_text_embeddings

    with patch("api.services.rag_pipeline.upsert_embeddings") as mock_upsert:
        df = pd.DataFrame({'type': ['mosquito bites'], 'chunk': ['foo'], 'embedding': [[0.1, 0.2]]})
        load_text_embeddings(df, collection=None, batch_size=1)
        mock_upsert.assert_called()


def test_chunk_char_split():
    from api.services.rag_pipeline import chunk

    with patch("api.services.rag_pipeline.glob.glob", return_value=["/tmp/test.txt"]):
        with patch("api.services.rag_pipeline.os.path.basename", return_value="mosquito bites.txt"):
            with patch("api.services.rag_pipeline.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = "test text"
                with patch("api.services.rag_pipeline.CharacterTextSplitter") as mock_splitter:
                    mock_splitter.return_value.create_documents.return_value = [
                        MagicMock(page_content="chunk1"),
                        MagicMock(page_content="chunk2"),
                    ]
                    with patch("api.services.rag_pipeline.pd.DataFrame") as mock_df:
                        mock_df.return_value.shape = (2, 1)
                        mock_df.return_value.head.return_value = None
                        with patch("api.services.rag_pipeline.open", create=True) as mock_file:
                            chunk(method="char-split")
                            # Assert that DataFrame was called with correct chunks
                            mock_df.assert_called_with(["chunk1", "chunk2"], columns=["chunk"])
                            # Assert that file was opened for writing output
                            mock_file.assert_called()


def test_embed_char_split():
    from api.services.rag_pipeline import embed

    with patch("api.services.rag_pipeline.glob.glob", return_value=["/tmp/chunks-char-split-mosquito.jsonl"]):
        with patch("api.services.rag_pipeline.pd.read_json") as mock_read_json:
            mock_df = MagicMock()
            mock_df.shape = (2, 1)
            mock_df.head.return_value = None
            mock_col = MagicMock()
            mock_col.values = MagicMock()
            mock_col.values.tolist.return_value = ["chunk1", "chunk2"]
            mock_df.__getitem__.side_effect = lambda key: mock_col if key == "chunk" else []
            mock_read_json.return_value = mock_df
            with patch("api.services.rag_pipeline.generate_text_embeddings", return_value=[[0.1, 0.2], [0.1, 0.2]]):
                with patch("api.services.rag_pipeline.open", create=True):
                    with patch("api.services.rag_pipeline.time.sleep"):
                        embed(method="char-split")


def test_chunk_recursive_split():
    from api.services.rag_pipeline import chunk

    with patch("api.services.rag_pipeline.glob.glob", return_value=["/tmp/test.txt"]):
        with patch("api.services.rag_pipeline.os.path.basename", return_value="mosquito bites.txt"):
            with patch("api.services.rag_pipeline.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = "test text"
                with patch("api.services.rag_pipeline.RecursiveCharacterTextSplitter") as mock_splitter:
                    mock_splitter.return_value.create_documents.return_value = [
                        MagicMock(page_content="chunk1"),
                        MagicMock(page_content="chunk2"),
                    ]
                    with patch("api.services.rag_pipeline.pd.DataFrame") as mock_df:
                        mock_df.return_value.shape = (2, 1)
                        mock_df.return_value.head.return_value = None
                        with patch("api.services.rag_pipeline.open", create=True) as mock_file:
                            chunk(method="recursive-split")
                            mock_df.assert_called_with(["chunk1", "chunk2"], columns=["chunk"])
                            mock_file.assert_called()


def test_chunk_semantic_split():
    from api.services.rag_pipeline import chunk

    with patch("api.services.rag_pipeline.glob.glob", return_value=["/tmp/test.txt"]):
        with patch("api.services.rag_pipeline.os.path.basename", return_value="mosquito bites.txt"):
            with patch("api.services.rag_pipeline.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = "test text"
                with patch("api.services.rag_pipeline._get_semantic_chunker") as mock_semantic:
                    mock_chunker = MagicMock()
                    mock_chunker.create_documents.return_value = [MagicMock(page_content="chunk1"), MagicMock(page_content="chunk2")]
                    mock_semantic.return_value = MagicMock(return_value=mock_chunker)
                    with patch("api.services.rag_pipeline.pd.DataFrame") as mock_df:
                        mock_df.return_value.shape = (2, 1)
                        mock_df.return_value.head.return_value = None
                        with patch("api.services.rag_pipeline.open", create=True) as mock_file:
                            chunk(method="semantic-split")
                            mock_df.assert_called_with(["chunk1", "chunk2"], columns=["chunk"])
                            mock_file.assert_called()


def test_load_char_split():
    from api.services.rag_pipeline import load

    with patch("api.services.rag_pipeline.glob.glob", return_value=["/tmp/embeddings-char-split-mosquito.jsonl"]):
        with patch("api.services.rag_pipeline.pd.read_json") as mock_read_json:
            mock_df = MagicMock()
            mock_df.shape = (2, 1)
            mock_df.head.return_value = None
            mock_read_json.return_value = mock_df
            with patch("api.services.rag_pipeline.load_text_embeddings") as mock_load_text_embeddings:
                load(method="char-split")
                mock_load_text_embeddings.assert_called()
