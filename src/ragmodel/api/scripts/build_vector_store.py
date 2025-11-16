import os
import sys

# add project root to python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from api.services.rag_pipeline import chunk, embed, load
from api.scripts.create_index import create_index


if __name__ == "__main__":
    print("Building vector store…")
    
    create_index()

    chunk(method="char-split")
    embed(method="char-split")
    load(method="char-split")

    print("Done.")
