import json
from sentence_transformers import SentenceTransformer
import chromadb
import os

# Specify the directory where ChromaDB will store the database files
persist_directory = os.path.join("research", "chromaFiles")

# Initialize ChromaDB with persistent storage settings
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_or_create_collection("lyrics_similarity_collection")

# Check the collection count before querying
print(f"Number of documents in collection (should be >0 if data persists): {collection.count()}")

# Load the model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", trust_remote_code=True)
model.max_seq_length = 512
model.tokenizer.padding_side = "right"

# Load songs from every lyrics file, skipping songs with no lyrics and duplicate ids
json_file_paths = ['data/t_songs_with_lyrics.json', 'data/pop_songs_with_lyrics.json', 'data/spotify_songs_with_lyrics.json']
lyrics_data = []
seen_ids = set()
for json_file_path in json_file_paths:
    if not os.path.exists(json_file_path):
        print(f"Skipping missing file {json_file_path}")
        continue
    with open(json_file_path, 'r') as file:
        for song in json.load(file):
            if song.get('lyrics') and song['id'] not in seen_ids:
                seen_ids.add(song['id'])
                lyrics_data.append(song)
print(f"Loaded {len(lyrics_data)} songs with lyrics")

# Skip songs already stored so re-running only embeds new songs
stored_ids = set(collection.get(include=[])['ids'])
lyrics_data = [song for song in lyrics_data if str(song['id']) not in stored_ids]
print(f"{len(lyrics_data)} new songs to embed")

# Function to add end-of-sequence token if required
def add_eos(input_examples):
    eos_token = model.tokenizer.eos_token
    if eos_token is not None:
        input_examples = [input_example + eos_token for input_example in input_examples]
    return input_examples

# Embed and store in batches, which is much faster than one song at a time
BATCH_SIZE = 256
for start in range(0, len(lyrics_data), BATCH_SIZE):
    batch = lyrics_data[start:start + BATCH_SIZE]

    lyric_embeddings = model.encode(add_eos([song['lyrics'] for song in batch]), batch_size=32, normalize_embeddings=True)

    collection.add(
        embeddings=lyric_embeddings.tolist(),
        metadatas=[{
            "artist": song['artist'],
            "title": song['title'],
            "lyrics": song['lyrics'],
        } for song in batch],
        ids=[str(song['id']) for song in batch],
        documents=[""] * len(batch)
    )

    # Print confirmation
    print(f"Stored {start + len(batch)}/{len(lyrics_data)} songs in the Chroma database.")

print("All entries stored with similarity scores in the Chroma database.")

# Final check on the number of stored documents
stored_results = collection.count()
print(f"Number of stored documents in collection: {stored_results}")
