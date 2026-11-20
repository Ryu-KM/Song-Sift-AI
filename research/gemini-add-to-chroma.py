import json
import google.generativeai as genai
import os
import chromadb
import concurrent.futures

persist_directory = "research/chromaFiles"

# Initialize ChromaDB with persistent storage settings
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_or_create_collection("gemini_lyrics_collection", metadata={"hnsw:space": "cosine"})

# Check the collection count before querying
print(f"Number of documents in collection (should be >0 if data persists): {collection.count()}")

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load songs from every lyrics file, skipping songs with no lyrics and duplicate ids
json_file_paths = ['data/t_songs_with_lyrics.json', 'data/pop_songs_with_lyrics.json']
lyrics_data = []
seen_ids = set()
for json_file_path in json_file_paths:
    with open(json_file_path, 'r') as file:
        for song in json.load(file):
            if song.get('lyrics') and song['id'] not in seen_ids:
                seen_ids.add(song['id'])
                lyrics_data.append(song)
print(f"Loaded {len(lyrics_data)} songs with lyrics")

# Function to split lyrics into exactly 5 chunks at newline characters
def split_into_chunks(lyrics, num_chunks=5):
    """
    Splits lyrics into exactly 5 chunks, respecting newline boundaries.
    
    :param lyrics: The full lyrics as a string
    :param num_chunks: Number of chunks to create
    :return: List of chunked lyrics
    """
    lines = lyrics.strip().split("\n")  # Split lyrics by newlines
    total_lines = len(lines)

    if total_lines <= num_chunks:  
        # If fewer lines than chunks, return each line as a separate chunk
        return [line.strip() for line in lines]

    # Determine chunk sizes as evenly as possible
    chunk_size = total_lines // num_chunks
    remainder = total_lines % num_chunks

    chunks = []
    start = 0
    for i in range(num_chunks):
        extra = 1 if i < remainder else 0  # Distribute remaining lines evenly
        end = start + chunk_size + extra
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks

def embed_chunk(chunk):
    """
    Embeds a single chunk of lyrics using the Gemini API.
    
    :param chunk: The lyrics chunk to be embedded
    :return: The embedding for the chunk
    """
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        output_dimensionality=768,  # must match between ingest and query
        content=chunk
    )
    return result['embedding']

# Process each song
for song in lyrics_data:
    title = song['title']
    artist = song['artist']
    lyrics = song['lyrics']
    song_id = song['id']

    # Create exactly 5 lyric chunks
    lyric_chunks = split_into_chunks(lyrics, num_chunks=5)

    # map() keeps results in chunk order so chunk_index matches the lyrics position
    with concurrent.futures.ThreadPoolExecutor() as executor:
        embeddings = list(zip(lyric_chunks, executor.map(embed_chunk, lyric_chunks)))

    for i, (chunk, embedding) in enumerate(embeddings):
        # Store each chunk in ChromaDB
        collection.add(
            embeddings=[embedding],
            metadatas=[{
                "artist": artist, 
                "title": title, 
                "chunk_index": i,
                "lyrics_chunk": chunk,
            }],
            ids=[f"{song_id}_chunk_{i}"],
            documents=[chunk]
        )

    print(f"Stored '{title}' by {artist} in ChromaDB with {len(lyric_chunks)} chunks.")

# Final check on the number of stored documents
stored_results = collection.count()
print(f"Number of stored lyric chunks in collection: {stored_results}")
