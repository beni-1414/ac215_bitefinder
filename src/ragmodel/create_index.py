import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
name = os.environ.get("PINECONE_INDEX", "bugbite-rag")
cloud = os.environ.get("PINECONE_CLOUD", "gcp")
region = os.environ.get("PINECONE_REGION", "us-central1")

existing = [i.name for i in pc.list_indexes()]
if name in existing:
    print(f"Deleting existing index: {name}")
    pc.delete_index(name)
    
    # Wait for deletion to complete
    print("Waiting for deletion to complete...")
    while name in [i.name for i in pc.list_indexes()]:
        time.sleep(2)
    print("✓ Index deleted")

print(f"Creating index: {name} with dimension=768")
pc.create_index(
    name=name,
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(cloud=cloud, region=region),
)

# Wait for index to be ready
print("Waiting for index to be ready...")
while not pc.describe_index(name).status.ready:
    time.sleep(2)

print(f"✓ Index ready: {name} (dimension=768)")