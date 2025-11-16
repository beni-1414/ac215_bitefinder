# This test calls chat() directly, but mocks embeddings + Pinecone.
# tests/unit/test_rag_pipeline.py
from unittest.mock import patch
from api.services.rag_pipeline import chat


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


if __name__ == "__main__":
    test_chat_function_builds_payload_correctly()
    print("ALL TESTS PASSED")
