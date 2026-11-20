import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Specify the directory where ChromaDB has stored the database files
persist_directory = "research/chromaFiles"

# Initialize ChromaDB with persistent storage settings
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_or_create_collection("lyrics_similarity_collection", metadata={"hnsw:space": "cosine"})

# Load the model
model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)
model.max_seq_length = 512
model.tokenizer.padding_side = "right"

# Function to add end-of-sequence token if required
def add_eos(input_examples):
    eos_token = model.tokenizer.eos_token
    if eos_token is not None:
        input_examples = [input_example + eos_token for input_example in input_examples]
    return input_examples

# Embed the query text
query_text = "I robbed a bank and need a car to leave"  # Adjust to a common term or a known stored lyric snippet for testing
query_embedding = model.encode(add_eos([query_text]), batch_size=1, normalize_embeddings=True)
query_embedding = query_embedding.squeeze().tolist()  # Convert to list format

# Perform the query
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include= ["embeddings", "metadatas", "distances"]
)

print(f"Number of stored documents in collection: {collection.count()}")

print(results);

if results["documents"] and results["distances"] and results["embeddings"]:
    for i in range(len(results["documents"][0])):
        artist = results["metadatas"][0][i]["artist"]
        title = results["metadatas"][0][i]["title"]
        
        # Check if embedding is available for this result
        if results["embeddings"][0] is not None:
            retrieved_embedding = results["embeddings"][0][i]
            distance = results["distances"][0][i]  # Get the distance for this result
            
            # Calculate the similarity score assuming distance is cosine distance
            similarity_score = (1 - distance) * 100
            
            print(f"\nResult {i+1}:")
            print(f"Artist: {artist}")
            print(f"Title: {title}")
            print(f"Similarity Score: {similarity_score:.2f}%")
        else:
            print(f"No embedding returned for result {i+1}")
else:
    print("No results found. Try verifying stored data or adjusting the query.")