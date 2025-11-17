# This script handles the end-to-end retrieval-augmentation steps (chunking, embedding, and loading into Pinecone).

import os
from dotenv import load_dotenv, find_dotenv

import argparse
import pandas as pd
import json
import time
import glob
import hashlib

from google import genai
from google.genai import types
from google.genai import errors

# Langchain
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# pinecone
from api.services.pinecone_adapter import upsert_embeddings, query_by_vector


# load .env before importing anything that needs env vars
load_dotenv(find_dotenv())

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
GENERATIVE_MODEL = "gemini-2.0-flash-001"
EMBEDDING_DIMENSION = 768

INPUT_FOLDER = "input-datasets"
OUTPUT_FOLDER = "outputs"

#############################################################################
#                       Initialize the LLM Client (Vertex AI)               #
llm_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
#############################################################################

# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = (
    "You are an AI assistant specialized in bug bites. Your responses are based solely on the "
    "information provided in the text chunks given to you. Do not use any external knowledge or make "
    "assumptions beyond what is explicitly stated in these chunks.\n\n"
    "When answering a query:\n"
    "1. Carefully read all the text chunks provided.\n"
    "2. Identify the most relevant information from these chunks to address the user's question.\n"
    "3. Formulate your response using only the information found in the given chunks.\n"
    "4. If the provided chunks do not contain sufficient information to answer the query, state that you "
    "don't have enough information to provide a complete answer.\n"
    "5. Always maintain a professional and knowledgeable tone, befitting a bug bite expert.\n"
    "6. If there are contradictions in the provided chunks, mention this in your response and explain the "
    "different viewpoints presented.\n\n"
    "Remember:\n"
    "- You are an expert in bug bite, but your knowledge is limited to the information in the provided chunks.\n"
    "- Do not invent information or draw from knowledge outside of the given text chunks.\n"
    "- If asked about topics unrelated to bug bites, politely redirect the conversation back to bug "
    "bite-related subjects.\n"
    "- Be concise in your responses while ensuring you cover all relevant information from the chunks.\n\n"
    "Your goal is to provide accurate, helpful information about bug bites based solely on the content of "
    "the text chunks you receive with each query."
)


bug_mappings = {
    "ant bites": {"bug": "ant"},
    "bed bug bites": {"bug": "bed bug"},
    "chigger bites": {"bug": "chigger"},
    "flea bites": {"bug": "flea"},
    "mosquito bites": {"bug": "mosquito"},
    "spider bites": {"bug": "spider"},
    "tick bites": {"bug": "tick"},
}


def _get_semantic_chunker():
    from ac215_bitefinder.src.ragmodel.api.services.semantic_splitter import SemanticChunker

    return SemanticChunker


# CHANGED: Use Vertex AI client instead of genai.embed_content
def generate_query_embedding(query: str):
    kwargs = {"output_dimensionality": EMBEDDING_DIMENSION}
    response = llm_client.models.embed_content(
        model=EMBEDDING_MODEL, contents=query, config=types.EmbedContentConfig(**kwargs)
    )
    return response.embeddings[0].values


# CHANGED: Use Vertex AI with batch processing
def generate_text_embeddings(
    chunks, dimensionality: int = 768, batch_size: int = 250, max_retries: int = 5, retry_delay: int = 5
):
    # Max batch size is 250 for Vertex AI
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        retry_count = 0
        while retry_count <= max_retries:
            try:
                response = llm_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=dimensionality),
                )
                all_embeddings.extend([embedding.values for embedding in response.embeddings])
                break

            except errors.APIError as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"Failed to embed after {max_retries} attempts: {e}")
                    raise

                wait_time = retry_delay * (2 ** (retry_count - 1))
                print(f"API error: {e.message}. Retrying in {wait_time}s (attempt {retry_count}/{max_retries})...")
                time.sleep(wait_time)

    return all_embeddings


# Pinecone section (unchanged)
def load_text_embeddings(df, collection, batch_size=500):
    df["id"] = df.index.astype(str)
    hashed_books = df["type"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
    df["id"] = hashed_books + "-" + df["id"]

    metadata = {"type": df["type"].tolist()[0]}
    if metadata["type"] in bug_mappings:
        bug_mapping = bug_mappings[metadata["type"]]
        metadata["bug"] = bug_mapping["bug"]

    total_inserted = 0
    for i in range(0, df.shape[0], batch_size):
        batch = df.iloc[i : i + batch_size].copy().reset_index(drop=True)

        ids = batch["id"].tolist()
        documents = batch["chunk"].tolist()
        metadatas = [metadata for _ in batch["type"].tolist()]
        embeddings = batch["embedding"].tolist()

        upsert_embeddings(
            ids=ids,
            texts=documents,
            metadatas=metadatas,
            vectors=embeddings,
        )

        total_inserted += len(batch)
        print(f"Inserted {total_inserted} items...")

    print(f"Finished inserting {total_inserted} items for type '{metadata['type']}'")


def chunk(method="char-split"):
    print("chunk()")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    text_files = glob.glob(os.path.join(INPUT_FOLDER, "bugs", "*.txt"))
    print("Number of files to process:", len(text_files))

    for text_file in text_files:
        print("Processing file:", text_file)
        filename = os.path.basename(text_file)
        bite_name = filename.split(".")[0]

        with open(text_file) as f:
            input_text = f.read()

        text_chunks = None
        if method == "char-split":
            chunk_size = 350
            chunk_overlap = 20
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="", strip_whitespace=False
            )
            text_chunks = text_splitter.create_documents([input_text])
            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        elif method == "recursive-split":
            chunk_size = 350
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
            text_chunks = text_splitter.create_documents([input_text])
            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        elif method == "semantic-split":
            SemanticChunker = _get_semantic_chunker()
            text_splitter = SemanticChunker(embedding_function=generate_text_embeddings)
            text_chunks = text_splitter.create_documents([input_text])
            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        if text_chunks is not None:
            data_df = pd.DataFrame(text_chunks, columns=["chunk"])
            data_df["type"] = bite_name
            print("Shape:", data_df.shape)
            print(data_df.head())

            jsonl_filename = os.path.join(OUTPUT_FOLDER, f"chunks-{method}-{bite_name}.jsonl")
            with open(jsonl_filename, "w") as json_file:
                json_file.write(data_df.to_json(orient="records", lines=True))


def embed(method="char-split"):
    print("embed()")

    jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"chunks-{method}-*.jsonl"))
    print("Number of files to process:", len(jsonl_files))

    for jsonl_file in jsonl_files:
        print("Processing file:", jsonl_file)

        data_df = pd.read_json(jsonl_file, lines=True)
        print("Shape:", data_df.shape)
        print(data_df.head())

        chunks = data_df["chunk"].values
        chunks = chunks.tolist()
        if method == "semantic-split":
            embeddings = generate_text_embeddings(chunks, EMBEDDING_DIMENSION, batch_size=15)
        else:
            embeddings = generate_text_embeddings(chunks, EMBEDDING_DIMENSION, batch_size=100)
        data_df["embedding"] = embeddings

        time.sleep(5)

        print("Shape:", data_df.shape)
        print(data_df.head())

        jsonl_filename = jsonl_file.replace("chunks-", "embeddings-")
        with open(jsonl_filename, "w") as json_file:
            json_file.write(data_df.to_json(orient="records", lines=True))


def load(method="char-split"):
    print("load()")

    jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"embeddings-{method}-*.jsonl"))
    print("Number of files to process:", len(jsonl_files))

    total = 0
    for jsonl_file in jsonl_files:
        print("Processing file:", jsonl_file)
        data_df = pd.read_json(jsonl_file, lines=True)
        print("Shape:", data_df.shape)
        print(data_df.head())

        load_text_embeddings(data_df, collection=None)
        total += data_df.shape[0]

    print(f"Finished inserting {total} total items into pinecone index '{os.getenv('PINECONE_INDEX', '')}'")


def query(method="char-split"):
    print("query()")

    query = "Prevention for mosquito bites?"
    query_embedding = generate_query_embedding(query)

    search_string = "prevent"

    results = query_by_vector(
        query_vec=query_embedding,
        top_k=10,
        metadata_filter={"type": {"$eq": "mosquito bites"}},
        contains_substring=search_string,
    )

    print("Query:", query)
    print("\n\nResults:", results)


def _normalize_conf_to_percent(conf: float) -> float:
    pct = conf * 100.0 if 0.0 <= conf <= 1.0 else conf
    try:
        pct = float(pct)
    except Exception:
        pct = 0.0
    return max(0.0, min(100.0, pct))


# changed to stop before LLM output
def chat(method="char-split", symptoms: str = "", conf: float = 0.0, bug_class: str = ""):
    print("chat()")

    user_question = f"How can I treat {bug_class} bites? My symptoms: {symptoms}"
    query_embedding = generate_query_embedding(user_question)
    print("Query:", user_question, "\n")

    results = query_by_vector(
        query_vec=query_embedding,
        top_k=10,
        metadata_filter={"bug": {"$eq": bug_class}},
    )

    conf_pct = _normalize_conf_to_percent(conf)
    prompt = (
        f"A patient reported experiencing the following: {symptoms}.\n"
        f"We believe with {conf_pct:.1f}% confidence that they have a {bug_class} bug bite.\n"
        "What do you recommend they do to treat it?"
    )

    context = "\n".join(results["documents"][0])

    # instead of calling the LLM, build a payload and return it
    payload = {"question": user_question, "prompt": prompt, "context": context, "bug_class": bug_class}

    print("Prepared payload (pre-LLM):", json.dumps(payload, indent=2))
    return payload


# CHANGED: Use Vertex AI for generation
def agent(
    method="char-split",
    question: str = "Describe how I can prevent mosquito bites?",
    where: dict | None = {"type": {"$eq": "mosquito bites"}},
    contains: str | None = None,
    top_k: int = 10,
):
    print("agent()")

    qvec = generate_query_embedding(question)

    results = query_by_vector(
        query_vec=qvec,
        top_k=top_k,
        metadata_filter=where,
        contains_substring=contains,
    )

    chunks = results.get("documents", [[]])[0]
    if not chunks:
        print("no chunks retrieved; try relaxing filters")
        return None

    context = "\n\n".join(chunks)

    prompt = (
        f"{SYSTEM_INSTRUCTION.strip()}\n\n"
        f"user question:\n{question}\n\n"
        f"context chunks:\n{context}\n\n"
        "answer using only the context above. if info is missing, say you do not have enough information."
    )

    # return payload instead of calling the model
    payload = {"question": question, "prompt": prompt, "context": context}

    print("Prepared payload (pre-LLM):", json.dumps(payload, indent=2))
    return payload


def main(args=None):
    print("CLI Arguments:", args)

    if args.chunk:
        chunk(method=args.chunk_type)

    if args.embed:
        embed(method=args.chunk_type)

    if args.load:
        load(method=args.chunk_type)

    if args.query:
        query(method=args.chunk_type)

    if args.chat:
        symptoms = args.symptoms
        conf = args.conf
        bug_class = args.vision_class
        chat(method=args.chunk_type, symptoms=symptoms, conf=conf, bug_class=bug_class)

    if args.agent:
        agent(method=args.chunk_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI")

    parser.add_argument("--chunk", action="store_true", help="Chunk text")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings")
    parser.add_argument("--load", action="store_true", help="Load embeddings to vector db")
    parser.add_argument("--query", action="store_true", help="Query vector db")
    parser.add_argument("--chat", action="store_true", help="Chat with LLM")
    parser.add_argument("--agent", action="store_true", help="Chat with LLM Agent")
    parser.add_argument("--chunk_type", default="char-split", help="char-split | recursive-split | semantic-split")
    parser.add_argument("--symptoms", type=str, help="User symptom text")
    parser.add_argument("--conf", type=float, help="Vision model confidence rate")
    parser.add_argument("--class", dest="vision_class", type=str, help="Bug class, e.g., mosquito")

    args = parser.parse_args()
    main(args)
