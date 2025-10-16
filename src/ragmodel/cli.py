import os
import argparse
import pandas as pd
import json
import time
import glob
import hashlib
import chromadb

# Vertex AI
from google import genai
from google.genai import types
from google.genai.types import Content, Part, GenerationConfig, ToolConfig
from google.genai import errors

# Langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from semantic_splitter import SemanticChunker
import agent_tools

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
GENERATIVE_MODEL = "gemini-2.0-flash-001"
INPUT_FOLDER = "input-datasets"
OUTPUT_FOLDER = "outputs"
CHROMADB_HOST = "llm-rag-chromadb"
CHROMADB_PORT = 8000

#############################################################################
#                       Initialize the LLM Client                           #
llm_client = genai.Client(
    vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
#############################################################################

# Initialize the GenerativeModel with specific system instructions
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
bug_mappings = {
    "ant bites": {"bug": "ant"},#, "year": 2022},
    "bed bug bites": {"bug": "bed bug"},
    "chigger bites": {"bug": "chigger"},
    "flea bites": {"bug": "flea"},
    "mosquito bites": {"bug": "mosquito"},
    "spider bites": {"bug": "spider"},
    "tick bites": {"bug": "tick"}
}


def generate_query_embedding(query):
    kwargs = {
        "output_dimensionality": EMBEDDING_DIMENSION
    }
    response = llm_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(**kwargs)
    )
    return response.embeddings[0].values


def generate_text_embeddings(chunks, dimensionality: int = 256, batch_size=250, max_retries=5, retry_delay=5):
    # Max batch size is 250 for Vertex AI
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        # Retry logic with exponential backoff
        retry_count = 0
        while retry_count <= max_retries:
            try:
                response = llm_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=dimensionality),
                )
                all_embeddings.extend(
                    [embedding.values for embedding in response.embeddings])
                break

            except errors.APIError as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(
                        f"Failed to generate embeddings after {max_retries} attempts. Last error: {str(e)}")
                    raise

                # Calculate delay with exponential backoff
                wait_time = retry_delay * (2 ** (retry_count - 1))
                print(
                    f"API error (code: {e.code}): {e.message}. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})...")
                time.sleep(wait_time)

    return all_embeddings

# book --> 'type' of bite, author --> bug
def load_text_embeddings(df, collection, batch_size=500):

    # Generate ids
    df["id"] = df.index.astype(str)
    hashed_books = df["type"].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
    df["id"] = hashed_books + "-" + df["id"]

    metadata = {
        "type": df["type"].tolist()[0] 
    }
    if metadata["type"] in bug_mappings:
        bug_mapping = bug_mappings[metadata["type"]]
        metadata["bug"] = bug_mapping["bug"]

    # Process data in batches
    total_inserted = 0
    for i in range(0, df.shape[0], batch_size):
        # Create a copy of the batch and reset the index
        batch = df.iloc[i:i+batch_size].copy().reset_index(drop=True)

        ids = batch["id"].tolist()
        documents = batch["chunk"].tolist()
        metadatas = [metadata for item in batch["type"].tolist()]
        embeddings = batch["embedding"].tolist()

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        total_inserted += len(batch)
        print(f"Inserted {total_inserted} items...")

    print(
        f"Finished inserting {total_inserted} items into collection '{collection.name}'")

# book_name --> bite_name
def chunk(method="char-split"):
    print("chunk()")

    # Make dataset folders
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get the list of text file
    text_files = glob.glob(os.path.join(INPUT_FOLDER, "bugs", "*.txt"))
    print("Number of files to process:", len(text_files))

    # Process     
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
            # Init the splitter
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator='', strip_whitespace=False)

            # Perform the splitting
            text_chunks = text_splitter.create_documents([input_text])
            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        elif method == "recursive-split":
            chunk_size = 350
            # Init the splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size)

            # Perform the splitting
            text_chunks = text_splitter.create_documents([input_text])
            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        elif method == "semantic-split":
            # Init the splitter
            text_splitter = SemanticChunker(
                embedding_function=generate_text_embeddings)
            # Perform the splitting
            text_chunks = text_splitter.create_documents([input_text])

            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        if text_chunks is not None:
            # Save the chunks
            data_df = pd.DataFrame(text_chunks, columns=["chunk"])
            data_df["type"] = bite_name
            print("Shape:", data_df.shape)
            print(data_df.head())

            jsonl_filename = os.path.join(
                OUTPUT_FOLDER, f"chunks-{method}-{bite_name}.jsonl")
            with open(jsonl_filename, "w") as json_file:
                json_file.write(data_df.to_json(orient='records', lines=True))


def embed(method="char-split"):
    print("embed()")

    # Get the list of chunk files
    jsonl_files = glob.glob(os.path.join(
        OUTPUT_FOLDER, f"chunks-{method}-*.jsonl"))
    print("Number of files to process:", len(jsonl_files))

    # Process
    for jsonl_file in jsonl_files:
        print("Processing file:", jsonl_file)

        data_df = pd.read_json(jsonl_file, lines=True)
        print("Shape:", data_df.shape)
        print(data_df.head())

        chunks = data_df["chunk"].values
        chunks = chunks.tolist()
        if method == "semantic-split":
            embeddings = generate_text_embeddings(
                chunks, EMBEDDING_DIMENSION, batch_size=15)
        else:
            embeddings = generate_text_embeddings(
                chunks, EMBEDDING_DIMENSION, batch_size=100)
        data_df["embedding"] = embeddings

        time.sleep(5)

        # Save
        print("Shape:", data_df.shape)
        print(data_df.head())

        jsonl_filename = jsonl_file.replace("chunks-", "embeddings-")
        with open(jsonl_filename, "w") as json_file:
            json_file.write(data_df.to_json(orient='records', lines=True))


def load(method="char-split"):
    print("load()")

    # Clear Cache
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-collection"
    print("Creating collection:", collection_name)

    try:
        # Clear out any existing items in the collection
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except Exception:
        print(f"Collection '{collection_name}' did not exist. Creating new.")

    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"})
    print(f"Created new empty collection '{collection_name}'")
    print("Collection:", collection)

    # Get the list of embedding files
    jsonl_files = glob.glob(os.path.join(
        OUTPUT_FOLDER, f"embeddings-{method}-*.jsonl"))
    print("Number of files to process:", len(jsonl_files))

    # Process
    for jsonl_file in jsonl_files:
        print("Processing file:", jsonl_file)

        data_df = pd.read_json(jsonl_file, lines=True)
        print("Shape:", data_df.shape)
        print(data_df.head())

        # Load data
        load_text_embeddings(data_df, collection)


def query(method="char-split"):
    print("load()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-collection"

    query = "Prevention for mosquito bites?" #"How is tolminc cheese made?"
    query_embedding = generate_query_embedding(query)
    #print("Embedding values:", query_embedding) ## silence this for now

    # Get the collection
    collection = client.get_collection(name=collection_name)

    # # 1: Query based on embedding value
    # results = collection.query(
    # 	query_embeddings=[query_embedding],
    # 	n_results=10
    # )
    # print("Query:", query)
    # print("\n\nResults:", results)

    # # 2: Query based on embedding value + metadata filter
    # results = collection.query(
    # 	query_embeddings=[query_embedding],
    # 	n_results=10,
    # 	where={"book":"The Complete Book of Cheese"}
    # )
    # print("Query:", query)
    # print("\n\nResults:", results)

    # # 3: Query based on embedding value + lexical search filter
    # search_string = "Italian"
    # results = collection.query(
    # 	query_embeddings=[query_embedding],
    # 	n_results=10,
    # 	where_document={"$contains": search_string}
    # )
    # print("Query:", query)
    # print("\n\nResults:", results)

    # 4: Query based on embedding value + lexical search filter
    search_string = "prevent"#"Italian"
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={"type": "mosquito bites"},
        where_document={"$contains": search_string}
    )
    print("Query:", query)
    print("\n\nResults:", results)

###
def _normalize_conf_to_percent(conf: float) -> float:
    """Accept 0–1 or 0–100. Return 0–100 clipped."""
    pct = conf * 100.0 if 0.0 <= conf <= 1.0 else conf
    try:
        pct = float(pct)
    except Exception:
        pct = 0.0
    return max(0.0, min(100.0, pct))

def chat(method="char-split", symptoms: str = "", conf: float = 0.0, bug_class: str = ""):
    print("chat()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-collection"

    #query = "How can I treat mosquito bites?"#"How is cheese made?"
    #query_embedding = generate_query_embedding(query)
    #print("Query:", query, "\n")

    ###
    user_question = f"How can I treat {bug_class} bites? My symptoms: {symptoms}"
    query_embedding = generate_query_embedding(user_question)
    print("Query:", user_question, "\n")

    #print("Embedding values:", query_embedding) ## silence for now
    # Get the collection
    collection = client.get_collection(name=collection_name)

    # Query based on embedding value
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where = {"bug": bug_class}
    )
    #print("\n\nResults:", results) ## silence for now

    #print(len(results["documents"][0])) ## silence for now

    conf_pct = _normalize_conf_to_percent(conf)
    prompt = (
        f"A patient reported experiencing the following: {symptoms}.\n"
        f"We believe with {conf_pct:.1f}% confidence that they have a {bug_class} bug bite.\n"
        "What do you recommend they do to treat it?"
    )

    INPUT_PROMPT = f"{prompt}\n" + "\n".join(results["documents"][0])
    #INPUT_PROMPT = f"""
	#{query}
	#{"\n".join(results["documents"][0])}
	#"""

    #print("INPUT_PROMPT: ", INPUT_PROMPT)  ## silence for now
    response = llm_client.models.generate_content(
        model=GENERATIVE_MODEL, contents=INPUT_PROMPT
    )
    generated_text = response.text

    print("LLM Response:", generated_text)


def get(method="char-split"):
    print("get()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-collection"

    # Get the collection
    collection = client.get_collection(name=collection_name)

    # Get documents with filters
    results = collection.get(
        where={"type": "mosquito"},
        limit=10
    )
    print("\n\nResults:", results)


def agent(method="char-split"):
    print("agent()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-collection"
    # Get the collection
    collection = client.get_collection(name=collection_name)

    # User prompt  ######ccccccc
    user_prompt_content = Content(
        role="user",
        parts=[
            #Part(text="Describe where cheese making is important in Pavlos's book?"),
            Part(text="Describe how I can prevent mosquito bites?"),
        ],
    )

    # Step 1: Prompt LLM to find the tool(s) to execute to find the relevant chunks in vector db
    print("user_prompt_content: ", user_prompt_content)
    response = llm_client.models.generate_content(
        model=GENERATIVE_MODEL,
        contents=user_prompt_content,
        config=types.GenerateContentConfig(
            temperature=0,
            tools=[agent_tools.cheese_expert_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="any"
                )
            )
        )
    )
    print("LLM Response:", response)

    # Step 2: Execute the function and send chunks back to LLM to answer get the final response
    function_calls = [part.function_call for part in response.candidates[0].content.parts if part.function_call]
    print("Function calls:", function_calls)
    function_responses = agent_tools.execute_function_calls(
        function_calls, collection, embed_func=generate_query_embedding)
    if len(function_responses) == 0:
        print("Function calls did not result in any responses...")
    else:
        # Call LLM with retrieved responses
        response = llm_client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=[
                user_prompt_content,  # User prompt
                response.candidates[0].content,  # Function call response
                Content(
                    parts=function_responses
                ),
            ],
            config=types.GenerateContentConfig(
                tools=[agent_tools.cheese_expert_tool]
            )
        )
        print("LLM Response:", response)


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
        ###
        symptoms = args.symptoms
        conf = args.conf
        bug_class = args.vision_class
        chat(method=args.chunk_type,
            symptoms=symptoms, 
            conf=conf, 
            bug_class=bug_class)

    if args.get:
        get(method=args.chunk_type)

    if args.agent:
        agent(method=args.chunk_type)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal '--help', it will provide the description
    parser = argparse.ArgumentParser(description="CLI")

    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Chunk text",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Generate embeddings",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load embeddings to vector db",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Query vector db",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Chat with LLM",
    )
    parser.add_argument(
        "--get",
        action="store_true",
        help="Get documents from vector db",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Chat with LLM Agent",
    )
    parser.add_argument("--chunk_type", default="char-split",
                        help="char-split | recursive-split | semantic-split")
    
    ###
    parser.add_argument("--symptoms", type=str, help="User symptom text")
    parser.add_argument("--conf", type=float, help="Vision model confidence rate")
    parser.add_argument("--class", dest="vision_class", type=str, help="Bug class, e.g., mosquito")

    


    args = parser.parse_args()

    main(args)
