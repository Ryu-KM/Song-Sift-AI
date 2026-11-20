from fastapi import FastAPI, File, HTTPException, Query, Body, UploadFile
from pydantic import BaseModel
import google.generativeai as genai
import chromadb
import concurrent.futures
from typing import List, Dict
import shutil
import tempfile
from audio_utils import process_snippet, process_chroma_snippet
import os
from utils import transcribe_audio_blob_to_text, convert_audio_to_wav

# FastAPI app setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI()

persist_directory = os.path.join("..", "..", "research", "chromaFiles")

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_or_create_collection("gemini_lyrics_collection", metadata={"hnsw:space": "cosine"})

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Executor for async operations
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Pydantic models for response structure
class LyricSearchResult(BaseModel):
    artist: str
    title: str
    similarity_score: float
    lyric_sections: List[str]  # List of lyric sections for this song
    youtube_query: str
    summary: str

class QueryResponse(BaseModel):
    query: str
    results: List[LyricSearchResult]

class SummaryRequest(BaseModel):
    artist: str
    title: str
    lyrics: List[str]

# Helper functions
def compute_query_embedding(query_text: str) -> List[float]:
    # Use Gemini API to get embedding for the query text
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        output_dimensionality=768,  # must match between ingest and query
        content=query_text
    )
    query_embedding = result['embedding']
    return query_embedding

def generate_summary(artist: str, title: str, lyrics: List[str]) -> str:
    try:
        model = genai.GenerativeModel("gemini-3.5-flash-lite")
        content_to_summarize = f"Song: {title} by {artist}. Lyrics:\n" + "\n".join(lyrics)
        response = model.generate_content(
            f"Summarize the following song in 2-3 sentences, always refer to the singers with their name and not just the singer:\n\n{content_to_summarize}"
        )

        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        return "No summary available."
    except Exception as e:
        print(f"Error generating summary for {title} by {artist}: {e}")
        return "Summary unavailable."

@app.post("/generate_summary")
def generate_summary_endpoint(summary_request: SummaryRequest = Body(...)):
    summary = generate_summary(summary_request.artist, summary_request.title, summary_request.lyrics)
    return {"summary": summary}  # Wrap in JSON

@app.post("/transcribe_audio")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        audio_blob = await file.read()
        text = transcribe_audio_blob_to_text(audio_blob)
        print(text)
        return {"transcription": text}
    except Exception as e:
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

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
        n_results=30,
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
                    youtube_query=youtube_query,
                    summary="" 
                )
            else:
                # If already exists, append the new lyric section and update similarity score
                grouped_results[key].lyric_sections.append(lyric_section)
                grouped_results[key].similarity_score = max(grouped_results[key].similarity_score, similarity_score)

        # Generate summaries
        ####
        #for key, song_result in grouped_results.items():

        #    song_result.summary = generate_summary(song_result.artist, song_result.title, song_result.lyric_sections)
        ####
        # Convert grouped results to a list
        response = list(grouped_results.values())
        return QueryResponse(query=query, results=response)
    
    else:
        raise HTTPException(status_code=404, detail="No results found. Try verifying stored data or adjusting the query.")
    
@app.post("/match_snippet")
def match_snippet_route(file: UploadFile = File(...)):
    try:
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Ensure the file is in .wav format
        if not tmp_path.endswith(".wav"):
            wav_path = tmp_path + ".wav"
            converted_path = convert_audio_to_wav(tmp_path, wav_path)
            if converted_path is None:
                raise HTTPException(status_code=500, detail="Failed to convert audio to .wav format.")
            tmp_path = converted_path

        db_name = "../../audio_features.db"
        result = process_snippet(tmp_path, db_name)

        print(f"Best match ID: {result['best_match_id']}")

        if result["filename"] is None:
            raise HTTPException(status_code=404, detail="No match found.")

        return {
            "match_filename": result["filename"],
            "match_distance": result["distance"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        # Clean up temporary files
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
