import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import argparse
import pandas as pd
import json
import time
import glob
import hashlib

import google.generativeai as genai

# Langchain text splitters
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import agent_tools

# Pinecone adapter - handles vector database operations
from pinecone_adapter import upsert_embeddings, query_by_vector

# Configure Google Generative AI (Gemini) with API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Model configuration
EMBEDDING_MODEL = "text-embedding-004"      # Google's text embedding model
GENERATIVE_MODEL = "models/gemini-2.5-flash"  # Gemini model for text generation
EMBEDDING_DIMENSION = 768                   # Must match Pinecone index dimension

# File paths
INPUT_FOLDER = "input-datasets"
OUTPUT_FOLDER = "outputs"

# System instruction for the LLM - defines its role and constraints
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in bug bites. Your responses are based solely on the information provided in the text chunks given to you. Do not use any external knowledge or make assumptions beyond what is explicitly stated in these chunks.

When answering a query:
1. Carefully read all the text chunks provided.
2. Identify the most relevant information from these chunks to address the user's question.
3. Formulate your response using only the information found in the given chunks.
4. If the provided chunks do not contain sufficient information to answer the query, state that you don't have enough information to provide a complete answer.
5. Always maintain a professional and knowledgeable tone, befitting a bug bite expert.
6. If there are contradictions in the provided chunks, mention this in your response and explain the different viewpoints presented.

Remember:
- You are an expert in bug bite, but your knowledge is limited to the information in the provided chunks.
- Do not invent information or draw from knowledge outside of the given text chunks.
- If asked about topics unrelated to bug bites, politely redirect the conversation back to bug bite-related subjects.
- Be concise in your responses while ensuring you cover all relevant information from the chunks.

Your goal is to provide accurate, helpful information about bug bites based solely on the content of the text chunks you receive with each query.
"""

# Metadata mappings for bug types
# Maps file names to canonical bug labels used in Pinecone metadata filtering
bug_mappings = {
    "ant bites": {"bug": "ant"},
    "bed bug bites": {"bug": "bed bug"},
    "chigger bites": {"bug": "chigger"},
    "flea bites": {"bug": "flea"},
    "mosquito bites": {"bug": "mosquito"},
    "spider bites": {"bug": "spider"},
    "tick bites": {"bug": "tick"}
}

# Initialize the generative model with system instructions
gen_model = genai.GenerativeModel(
    model_name=GENERATIVE_MODEL,
    system_instruction=SYSTEM_INSTRUCTION,
)

### helper functions

def _get_semantic_chunker():
    """Lazy import to avoid hard dependency when not using semantic splitting."""
    from semantic_splitter import SemanticChunker
    return SemanticChunker


def _normalize_conf_to_percent(conf: float) -> float:
    """
    Normalize confidence value to percentage (0-100).
    Accepts both 0-1 range and 0-100 range, returns 0-100 clipped.
    """
    pct = conf * 100.0 if 0.0 <= conf <= 1.0 else conf
    try:
        pct = float(pct)
    except Exception:
        pct = 0.0
    return max(0.0, min(100.0, pct))


### embedding functions

def generate_query_embedding(query: str):
    """
    Generate embeddings for search queries.
    
    Uses Google's text-embedding-004 model with task_type="retrieval_query"
    which optimizes embeddings for similarity search queries.
    
    Returns:
        List of floats: 768-dimensional embedding vector
    """
    res = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query",
        output_dimensionality=EMBEDDING_DIMENSION,
    )
    return res["embedding"]


def generate_text_embeddings(chunks, dimensionality: int = 768, batch_size: int = 250,
                             max_retries: int = 5, retry_delay: int = 5):
    """
    Generate embeddings for document chunks.
    
    Uses Google's text-embedding-004 model with task_type="retrieval_document"
    which optimizes embeddings for documents to be retrieved.
    
    Note: Unlike ChromaDB which can batch embed automatically, process
    chunks one at a time here with retry logic for robustness.
    
    Args:
        chunks: List of text strings to embed
        dimensionality: Output dimension (must match Pinecone index)
        batch_size: Number of chunks to process before sleeping
        max_retries: Maximum retry attempts on failure
        retry_delay: Initial delay between retries (exponential backoff)
    
    Returns:
        List of embeddings (768-dimensional vectors)
    """
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        for text in batch:
            retries = 0
            while True:
                try:
                    res = genai.embed_content(
                        model=EMBEDDING_MODEL,
                        content=text,
                        task_type="retrieval_document",
                        output_dimensionality=EMBEDDING_DIMENSION,
                    )
                    all_embeddings.append(res["embedding"])
                    break
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        print(f"Failed to embed after {max_retries} attempts: {e}")
                        raise
                    wait = retry_delay * (2 ** (retries - 1))
                    print(f"Embed error: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
    return all_embeddings


### pinecone data loading

def load_text_embeddings(df, collection, batch_size=500):
    """
    Load embeddings into Pinecone vector database.
    
    PINECONE vs CHROMADB DIFFERENCES:
    
    1. DATA STRUCTURE:
       - ChromaDB: Separate fields for ids, documents, embeddings, metadatas
         collection.add(ids=[], documents=[], embeddings=[], metadatas=[])
       
       - Pinecone: Unified vector format with metadata containing text
         upsert([{"id": ..., "values": [...], "metadata": {"text": ...}}])
    
    2. TEXT STORAGE:
       - ChromaDB: Has dedicated "documents" field for full text
       - Pinecone: Must store text in metadata["text"] field
    
    3. CONNECTION:
       - ChromaDB: Pass collection object
       - Pinecone: Uses global index from environment variables
    
    4. UPSERT vs ADD:
       - ChromaDB: collection.add() throws error on duplicate IDs
       - Pinecone: upsert() updates existing or creates new (idempotent)
    
    Args:
        df: DataFrame with columns: id, chunk, type, embedding
        collection: Unused (kept for API compatibility)
        batch_size: Number of vectors to upsert at once
    """
    # Generate unique IDs using hash of type + index
    df["id"] = df.index.astype(str)
    hashed_books = df["type"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
    df["id"] = hashed_books + "-" + df["id"]

    # Build shared metadata for all chunks from this file
    metadata = {"type": df["type"].tolist()[0]}
    if metadata["type"] in bug_mappings:
        bug_mapping = bug_mappings[metadata["type"]]
        metadata["bug"] = bug_mapping["bug"]

    # Process data in batches
    total_inserted = 0
    for i in range(0, df.shape[0], batch_size):
        batch = df.iloc[i:i+batch_size].copy().reset_index(drop=True)

        ids = batch["id"].tolist()
        documents = batch["chunk"].tolist()
        # Replicate the same metadata dict for each vector
        metadatas = [metadata for _ in batch["type"].tolist()]
        embeddings = batch["embedding"].tolist()

        # Call Pinecone adapter to upsert vectors
        # This replaces ChromaDB's collection.add()
        upsert_embeddings(
            ids=ids,
            texts=documents,      # Will be stored in metadata["text"]
            metadatas=metadatas,  # Additional metadata for filtering
            vectors=embeddings,   # 768-dimensional vectors
        )

        total_inserted += len(batch)
        print(f"Inserted {total_inserted} items...")

    print(f"Finished inserting {total_inserted} items for type '{metadata['type']}'")


### chunking functions

def chunk(method="char-split"):
    """
    Split input text files into smaller chunks.
    
    Supports three chunking methods:
    - char-split: Fixed character-based splitting with overlap
    - recursive-split: Recursive splitting that respects document structure
    - semantic-split: Semantic chunking based on meaning (uses embeddings)
    
    Outputs JSONL files with chunks for each bug type.
    """
    print("chunk()")

    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get list of bug bite text files
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
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap, 
                separator='', 
                strip_whitespace=False
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
            # Lazy import to avoid dependency when not using semantic splitting
            SemanticChunker = _get_semantic_chunker()
            text_splitter = SemanticChunker(embedding_function=generate_text_embeddings)
            text_chunks = text_splitter.create_documents([input_text])
            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        if text_chunks is not None:
            # Save chunks to JSONL file
            data_df = pd.DataFrame(text_chunks, columns=["chunk"])
            data_df["type"] = bite_name
            print("Shape:", data_df.shape)
            print(data_df.head())

            jsonl_filename = os.path.join(
                OUTPUT_FOLDER, f"chunks-{method}-{bite_name}.jsonl"
            )
            with open(jsonl_filename, "w") as json_file:
                json_file.write(data_df.to_json(orient='records', lines=True))


def embed(method="char-split"):
    """
    Generate embeddings for all chunks.
    
    Reads chunk JSONL files, generates embeddings using Google's
    text-embedding-004 model, and saves to new JSONL files.
    """
    print("embed()")

    # Get list of chunk files
    jsonl_files = glob.glob(os.path.join(
        OUTPUT_FOLDER, f"chunks-{method}-*.jsonl"
    ))
    print("Number of files to process:", len(jsonl_files))

    for jsonl_file in jsonl_files:
        print("Processing file:", jsonl_file)

        data_df = pd.read_json(jsonl_file, lines=True)
        print("Shape:", data_df.shape)
        print(data_df.head())

        chunks = data_df["chunk"].values.tolist()
        
        # Use smaller batch size for semantic splitting (more compute-intensive)
        if method == "semantic-split":
            embeddings = generate_text_embeddings(
                chunks, EMBEDDING_DIMENSION, batch_size=15
            )
        else:
            embeddings = generate_text_embeddings(
                chunks, EMBEDDING_DIMENSION, batch_size=100
            )
        
        data_df["embedding"] = embeddings

        time.sleep(5)  # Rate limiting

        # Save embeddings to new JSONL file
        print("Shape:", data_df.shape)
        print(data_df.head())

        jsonl_filename = jsonl_file.replace("chunks-", "embeddings-")
        with open(jsonl_filename, "w") as json_file:
            json_file.write(data_df.to_json(orient='records', lines=True))


def load(method="char-split"):
    """
    Load embeddings into Pinecone vector database.
    
    Reads embedding JSONL files and upserts all vectors into Pinecone.
    The Pinecone index must already exist with dimension=768.
    """
    print("load()")

    # Find all embedding files
    jsonl_files = glob.glob(os.path.join(
        OUTPUT_FOLDER, f"embeddings-{method}-*.jsonl"
    ))
    print("Number of files to process:", len(jsonl_files))

    total = 0
    for jsonl_file in jsonl_files:
        print("Processing file:", jsonl_file)
        data_df = pd.read_json(jsonl_file, lines=True)
        print("Shape:", data_df.shape)
        print(data_df.head())

        # Upsert to Pinecone
        load_text_embeddings(data_df, collection=None)
        total += data_df.shape[0]

    print(f"Finished inserting {total} total items into Pinecone index '{os.getenv('PINECONE_INDEX','')}'")


### Query functions

def query(method="char-split"):
    """
    Test query against Pinecone vector database.
    
    PINECONE vs CHROMADB QUERY DIFFERENCES:
    
    1. QUERY SYNTAX:
       - ChromaDB: collection.query(query_embeddings=[...], n_results=10, 
                                    where={...}, where_document={"$contains": "..."})
       - Pinecone: query_by_vector(query_vec=..., top_k=10, 
                                   metadata_filter={...}, contains_substring="...")
    
    2. METADATA FILTERING:
       - ChromaDB: where={"bug": "mosquito"}
       - Pinecone: metadata_filter={"bug": {"$eq": "mosquito"}}
    
    3. TEXT SEARCH:
       - ChromaDB: Built-in where_document={"$contains": "text"}
       - Pinecone: Custom post-processing with contains_substring parameter
    
    4. RESULTS FORMAT:
       - ChromaDB: {"ids": [[...]], "documents": [[...]], "metadatas": [[...]]}
       - Pinecone: Returns same format via adapter for compatibility
    """
    print("query()")

    query = "Prevention for mosquito bites?"
    query_embedding = generate_query_embedding(query)

    # Optional lexical filter (mimics ChromaDB's where_document)
    search_string = "prevent"

    results = query_by_vector(
        query_vec=query_embedding,
        top_k=10,
        metadata_filter={"type": {"$eq": "mosquito bites"}},
        contains_substring=search_string,
    )

    print("Query:", query)
    print("\n\nResults:", results)


def chat(method="char-split", symptoms: str = "", conf: float = 0.0, bug_class: str = ""):
    """
    Interactive chat with RAG (Retrieval-Augmented Generation).
    
    1. Embeds user question
    2. Retrieves relevant chunks from Pinecone filtered by bug type
    3. Constructs prompt with retrieved context
    4. Generates response using Gemini
    
    Args:
        method: Chunking method used
        symptoms: User-reported symptoms
        conf: Confidence score from vision model (0-1 or 0-100)
        bug_class: Bug type (e.g., "mosquito", "tick")
    """
    print("chat()")

    user_question = f"How can I treat {bug_class} bites? My symptoms: {symptoms}"
    query_embedding = generate_query_embedding(user_question)
    print("Query:", user_question, "\n")

    # Retrieve top matches filtered by bug type
    # Uses metadata filter on canonical "bug" field
    results = query_by_vector(
        query_vec=query_embedding,
        top_k=10,
        metadata_filter={"bug": {"$eq": bug_class}},
    )

    # Build prompt with context
    conf_pct = _normalize_conf_to_percent(conf)
    prompt = (
        f"A patient reported experiencing the following: {symptoms}.\n"
        f"We believe with {conf_pct:.1f}% confidence that they have a {bug_class} bug bite.\n"
        "What do you recommend they do to treat it?"
    )

    input_prompt = f"{prompt}\n" + "\n".join(results["documents"][0])

    # Generate response
    response = gen_model.generate_content(input_prompt)
    generated_text = response.text

    print("LLM Response:", generated_text)


def agent(method="char-split",
          question: str = "Describe how I can prevent mosquito bites?",
          where: dict | None = {"type": {"$eq": "mosquito bites"}},
          contains: str | None = None,
          top_k: int = 10):
    """
    Agent-based query with custom parameters.
    
    More flexible than chat() - allows custom filtering and retrieval parameters.
    
    Args:
        method: Chunking method
        question: User question
        where: Metadata filter dict (Pinecone format)
        contains: Optional substring to filter by
        top_k: Number of results to retrieve
    """
    print("agent()")

    # Embed question
    qvec = generate_query_embedding(question)

    # Retrieve from Pinecone with optional filters
    results = query_by_vector(
        query_vec=qvec,
        top_k=top_k,
        metadata_filter=where,
        contains_substring=contains,
    )

    # Build prompt from retrieved chunks
    chunks = results.get("documents", [[]])[0]
    if not chunks:
        print("No chunks retrieved; try relaxing filters")
        return
    
    context = "\n\n".join(chunks)

    prompt = (
        f"{SYSTEM_INSTRUCTION.strip()}\n\n"
        f"user question:\n{question}\n\n"
        f"context chunks:\n{context}\n\n"
        "answer using only the context above. if info is missing, say you do not have enough information."
    )

    # Generate answer
    resp = gen_model.generate_content(prompt)
    print("LLM Response:", resp.text)


### Main cli

def main(args=None):
    """Main entry point for CLI."""
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
    parser = argparse.ArgumentParser(description="BiteFinder RAG CLI")

    parser.add_argument("--chunk", action="store_true", help="Chunk text files")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings")
    parser.add_argument("--load", action="store_true", help="Load embeddings to Pinecone")
    parser.add_argument("--query", action="store_true", help="Test query against Pinecone")
    parser.add_argument("--chat", action="store_true", help="Chat with LLM using RAG")
    parser.add_argument("--agent", action="store_true", help="Agent-based query")
    parser.add_argument("--chunk_type", default="char-split",
                        help="Chunking method: char-split | recursive-split | semantic-split")
    parser.add_argument("--symptoms", type=str, help="User symptom text")
    parser.add_argument("--conf", type=float, help="Vision model confidence (0-1 or 0-100)")
    parser.add_argument("--class", dest="vision_class", type=str, 
                        help="Bug class from vision model (e.g., mosquito)")

    args = parser.parse_args()
    main(args)