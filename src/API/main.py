from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import concurrent.futures
from typing import List, Dict
import os

# FastAPI app setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI()

persist_directory = os.path.join("..", "..", "research", "chromaFiles")

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_or_create_collection("lyrics_similarity_collection", metadata={"hnsw:space": "cosine"})

# Load the model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", trust_remote_code=True)
model.max_seq_length = 512
model.tokenizer.padding_side = "right"

# Executor for async operations
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Pydantic models for response structure
class LyricSearchResult(BaseModel):
    artist: str
    title: str
    similarity_score: float
    lyric_sections: List[str]  # List of lyric sections for this song
    youtube_query: str

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
        n_results=20,
        include=["embeddings", "metadatas", "distances", "documents"]
    )

    if results["documents"] and results["distances"] and results["embeddings"]:
        grouped_results: Dict[str, LyricSearchResult] = {}

        # Iterate over results and group them by artist + title
        for i in range(len(results["documents"][0])):
            artist = results["metadatas"][0][i]["artist"]
            title = results["metadatas"][0][i]["title"]
            lyric_section = results["documents"][0][i]
            distance = results["distances"][0][i]
            similarity_score = (1 - (distance / 2)) * 100
            youtube_query = "https://www.youtube.com/results?search_query=" + artist + " - " + title

            # Use a unique key (artist + title) to group the results
            key = f"{artist}-{title}"
            
            if key not in grouped_results:
                grouped_results[key] = LyricSearchResult(
                    artist=artist,
                    title=title,
                    similarity_score=similarity_score,
                    lyric_sections=[lyric_section],  # Start a new list of lyric sections
                    youtube_query=youtube_query
                )
            else:
                # If already exists, append the new lyric section and update similarity score
                grouped_results[key].lyric_sections.append(lyric_section)
                grouped_results[key].similarity_score = max(grouped_results[key].similarity_score, similarity_score)

        # Convert grouped results to a list
        response = list(grouped_results.values())
        return QueryResponse(query=query, results=response)
    
    else:
        raise HTTPException(status_code=404, detail="No results found. Try verifying stored data or adjusting the query.")
