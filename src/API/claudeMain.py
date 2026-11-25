from fastapi import FastAPI, File, HTTPException, Query, Body, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import anthropic
import chromadb
import concurrent.futures
from typing import List, Dict
import shutil
import tempfile
import os
from audio_utils import process_snippet
from utils import transcribe_audio_blob_to_text, convert_audio_to_wav

# FastAPI app setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI()

# --- Vector DB -------------------------------------------------------------
# The Hugging Face collection built by research/add-to-chroma.py (~10k songs).
# It was created without a distance setting, so Chroma uses squared L2 distance.
persist_directory = os.getenv("CHROMA_PATH", os.path.join("..", "..", "research", "chromaFiles"))
os.makedirs(persist_directory, exist_ok=True)

client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_or_create_collection("lyrics_similarity_collection")

# --- Embeddings (local Hugging Face model; must match the one used at ingest) ---
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", trust_remote_code=True)
model.max_seq_length = 512
model.tokenizer.padding_side = "right"

# --- Summaries (Claude) ----------------------------------------------------
# anthropic.Anthropic() reads ANTHROPIC_API_KEY from the environment.
claude = anthropic.Anthropic()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
# Server-side refusal fallbacks are only supported on some models.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-2", "claude-fable-5-1", "claude-sonnet-5-5"}

# Executor for async operations
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

COSINE_BOTTOM_THRESHOLD = 0.1  # Unrelated songs
COSINE_TOP_THRESHOLD = 0.65     # Very similar songs

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
def add_user_friendly_score(similarity_score: float) -> int:
    """Need to modify the cosine similarity score to a more user-friendly score (0-100)."""
    scale = (similarity_score - COSINE_BOTTOM_THRESHOLD) / (COSINE_TOP_THRESHOLD - COSINE_BOTTOM_THRESHOLD)
    scale = max(0.0, min(1.0, scale))  # Clamp to [0, 1]
    return int(scale * 100)  # Scale to 0-100

def add_eos(input_examples):
    eos_token = model.tokenizer.eos_token
    if eos_token is not None:
        input_examples = [input_example + eos_token for input_example in input_examples]
    return input_examples

def compute_query_embedding(query_text: str) -> List[float]:
    query_embedding = model.encode(add_eos([query_text]), batch_size=1, normalize_embeddings=True)
    return query_embedding.squeeze().tolist()

def generate_summary(artist: str, title: str, lyrics: List[str]) -> str:
    content_to_summarize = f"Song: {title} by {artist}. Lyrics:\n" + "\n".join(lyrics)

    request = dict(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        output_config={"effort": "low"},  # short, simple task
        system=(
            "You summarize songs in 2-3 sentences. Always refer to the singers by name, "
            "never just 'the singer'. Reply with the summary only."
        ),
        messages=[{"role": "user", "content": content_to_summarize}],
    )
    if CLAUDE_MODEL in FALLBACK_MODELS:
        # If the model declines, the API retries on a fallback model in the same call.
        request["betas"] = ["server-side-fallback-2026-07-01"]
        request["fallbacks"] = "default"

    try:
        response = claude.beta.messages.create(**request)
    except anthropic.RateLimitError:
        print(f"Rate limited while summarizing {title} by {artist}")
        return "Summary unavailable (rate limited), try again shortly."
    except anthropic.APIStatusError as e:
        print(f"Claude API error {e.status_code} for {title} by {artist}: {e.message}")
        return "Summary unavailable."
    except anthropic.APIConnectionError:
        print(f"Could not reach the Claude API for {title} by {artist}")
        return "Summary unavailable."

    if response.stop_reason == "refusal":
        return "Summary unavailable."

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text or "No summary available."

@app.post("/generate_summary")
def generate_summary_endpoint(summary_request: SummaryRequest = Body(...)):
    summary = generate_summary(summary_request.artist, summary_request.title, summary_request.lyrics)
    return {"summary": summary}

@app.post("/transcribe_audio")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        audio_blob = await file.read()
        text = transcribe_audio_blob_to_text(audio_blob)
        return {"transcription": text}
    except Exception as e:
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/query_lyrics", response_model=QueryResponse)
def query_lyrics(
    query: str = Query(..., description="Query text to search for similar lyrics"),
    n_results: int = Query(50, ge=1, le=200, description="How many songs to return"),
):
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")

    # Compute the embedding for the query
    future = executor.submit(compute_query_embedding, query)
    query_embedding = future.result()

    # Query the Chroma DB for similar lyrics
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    if not results["metadatas"] or not results["metadatas"][0]:
        raise HTTPException(status_code=404, detail="No results found. Try verifying stored data or adjusting the query.")

    grouped_results: Dict[str, LyricSearchResult] = {}

    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        artist = metadata["artist"]
        title = metadata["title"]
        # This collection stores the full lyrics in metadata (documents are empty).
        lyric_section = metadata.get("lyrics", "")
        # Normalized embeddings + squared L2: cosine similarity = 1 - d/2
        similarity_score = add_user_friendly_score(1 - (distance / 2))
        youtube_query = "https://www.youtube.com/results?search_query=" + artist + " - " + title

        key = f"{artist}-{title}"
        if key not in grouped_results:
            grouped_results[key] = LyricSearchResult(
                artist=artist,
                title=title,
                similarity_score=similarity_score,
                lyric_sections=[lyric_section],
                youtube_query=youtube_query,
                summary="",
            )
        else:
            grouped_results[key].lyric_sections.append(lyric_section)
            grouped_results[key].similarity_score = max(grouped_results[key].similarity_score, similarity_score)

    return QueryResponse(query=query, results=list(grouped_results.values()))

@app.post("/match_snippet")
def match_snippet_route(file: UploadFile = File(...)):
    tmp_path = None
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

        db_name = os.getenv("AUDIO_DB_PATH", "../../audio_features.db")
        result = process_snippet(tmp_path, db_name)

        if result["filename"] is None:
            raise HTTPException(status_code=404, detail="No match found.")

        return {
            "match_filename": result["filename"],
            "match_distance": result["distance"],
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
