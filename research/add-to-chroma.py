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

# Load the lyrics from the JSON file (adjust the path to your file)
json_file_path = 'data/t_songs_with_lyrics.json'  # Change this to your JSON file path
with open(json_file_path, 'r') as file:
    lyrics_data = json.load(file)

# Function to add end-of-sequence token if required
def add_eos(input_examples):
    eos_token = model.tokenizer.eos_token
    if eos_token is not None:
        input_examples = [input_example + eos_token for input_example in input_examples]
    return input_examples

for song in lyrics_data:
    title = song['title']
    artist = song['artist']
    lyric = song['lyrics']
    song_id = song['id']
    
    lyric_embedding = model.encode(add_eos([lyric]), batch_size=1, normalize_embeddings=True)
    lyric_embedding = lyric_embedding.squeeze().tolist()

    collection.add(
        embeddings=[lyric_embedding],
        metadatas=[{
            "artist": artist, 
            "title": title, 
            "lyrics": lyric,
        }],
        ids=[str(song_id)],
        documents=[""]
    )

    # Print confirmation
    print(f"Stored '{title}' by {artist} in the Chroma database.")

print("All entries stored with similarity scores in the Chroma database.")

# Final check on the number of stored documents
stored_results = collection.count()
print(f"Number of stored documents in collection: {stored_results}")
