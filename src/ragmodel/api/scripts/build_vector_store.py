import os
import sys

# ALL imports must come before any other code
from api.services.rag_pipeline import chunk, embed, load
from api.scripts.create_index import create_index

# only after imports can we manipulate sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)


if __name__ == "__main__":
    print("Building vector store…")

    create_index()

    chunk(method="char-split")
    embed(method="char-split")
    load(method="char-split")

    print("Done.")
