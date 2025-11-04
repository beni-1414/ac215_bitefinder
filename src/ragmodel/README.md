# RAG model
This is used to chat with the LLM using the RAG system. This version on `pinecone` branch uses Pinecone (not ChromaDB)

To run on Vertex AI:
```
python cli-vertexAI.py --chunk --chunk_type char-split
python cli-vertexAI.py --embed --chunk_type char-split
python cli-vertexAI.py --load  --chunk_type char-split
python cli-vertexAI.py --query --chunk_type char-split
python cli-vertexAI.py --chat  --chunk_type char-split --symptoms "itchy ankles" --conf 0.9 --class mosquito
```

To run locally:
```
python cli.py --chunk --chunk_type char-split
python cli.py --embed --chunk_type char-split
python cli.py --load  --chunk_type char-split
python cli.py --query --chunk_type char-split
python cli.py --chat  --chunk_type char-split --symptoms "itchy ankles" --conf 0.9 --class mosquito
```

Also can change symptoms, conf, and class for diff results.