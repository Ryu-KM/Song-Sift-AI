import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import concurrent.futures
from typing import List

# Set environment variable to avoid warnings with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize FastAPI app
app = FastAPI()

# Directory where Chroma DB is persisted
persist_directory = os.path.join("research", "chromaFiles")

# Ensure the directory exists to prevent Chroma DB from being reinitialized
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Initialize Chroma client
client = chromadb.PersistentClient(path=persist_directory)

# Get or create a Chroma collection
collection = client.get_or_create_collection("lyrics_similarity_collection", metadata={"hnsw:space": "cosine"})

# Load the Sentence-Transformer model
model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)
model.max_seq_length = 512
model.tokenizer.padding_side = "right"

# Set up a ThreadPoolExecutor to handle embedding calculations concurrently
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Pydantic models for request/response validation
class LyricSearchResult(BaseModel):
    artist: str
    title: str
    similarity_score: float

class QueryResponse(BaseModel):
    query: str
    results: List[LyricSearchResult]

# Helper functions
def add_eos(input_examples):
    eos_token = model.tokenizer.eos_token
    if eos_token is not None:
        input_examples = [input_example + eos_token for input_example in input_examples]
    return input_examples

def compute_query_embedding(query_text: str) -> List[float]:
    query_embedding = model.encode(add_eos([query_text]), batch_size=1, normalize_embeddings=True)
    return query_embedding.squeeze().tolist()

# API route to handle lyric search queries
@app.get("/query_lyrics", response_model=QueryResponse)
def query_lyrics(query: str = Query(..., description="Query text to search for similar lyrics")):
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")
    
    # Compute the embedding for the query
    future = executor.submit(compute_query_embedding, query)
    query_embedding = future.result()

    # Query the Chroma DB for similar lyrics
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["embeddings", "metadatas", "distances", "documents"]
    )

    print("Results from Chroma DB:", results)

    # If results found, format them for the response
    if results["documents"] and results["distances"] and results["embeddings"]:
        response = []
        for i in range(len(results["documents"][0])):
            artist = results["metadatas"][0][i]["artist"]
            title = results["metadatas"][0][i]["title"]
            distance = results["distances"][0][i]
            similarity_score = (2 - distance) * 100  # Convert distance to similarity score
            response.append(LyricSearchResult(
                artist=artist,
                title=title,
                similarity_score=similarity_score
            ))
        return QueryResponse(query=query, results=response)
    else:
        raise HTTPException(status_code=404, detail="No results found. Try verifying stored data or adjusting the query.")
