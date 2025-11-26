import os
from typing import List, Dict, Any
from pinecone import Pinecone
from google.cloud import secretmanager


def get_secret(secret_id: str) -> str:
    """Retrieve a secret from Google Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.getenv('GCP_PROJECT')}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")


def _index():
    pinecone_api_key = get_secret("PINECONE_API_KEY")
    if not pinecone_api_key:
        pinecone_api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(os.environ["PINECONE_INDEX"])


def upsert_embeddings(
    ids: List[str],
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    vectors: List[List[float]],
):
    idx = _index()
    items = []
    for i in range(len(ids)):
        md = dict(metadatas[i]) if metadatas else {}
        md.setdefault("text", texts[i])  # store chunk text for retrieval
        items.append({"id": ids[i], "values": vectors[i], "metadata": md})
    idx.upsert(items)


def query_by_vector(
    query_vec: List[float],
    top_k: int = 10,
    metadata_filter: Dict[str, Any] | None = None,
    contains_substring: str | None = None,
):
    idx = _index()
    res = idx.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter or {},
    )
    docs, ids, mds = [], [], []
    for m in res.matches:
        md = m.metadata or {}
        txt = md.get("text", "")
        if contains_substring and contains_substring.lower() not in txt.lower():
            continue
        ids.append(m.id)
        docs.append(txt)
        mds.append(md)
    # return a minimal chroma-like shape so the rest of your code stays the same
    return {"ids": [ids], "documents": [docs], "metadatas": [mds]}
