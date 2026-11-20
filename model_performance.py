import os
import time
import requests
import json
import numpy as np
from sklearn.metrics import ndcg_score
import openai  # Import OpenAI API

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

GEMINI_API_URL = "http://localhost:8000/query_lyrics" 
HUGGINGFACE_API_URL = "http://localhost:8001/query_lyrics" 

K = 20
OUTPUT_FILE = "comparison_results.json"
QUERY_TEXT = "love songs"

def query_model(api_url, query):
    start_time = time.time()
    response = requests.get(api_url, params={"query": query})
    elapsed_time = time.time() - start_time
    
    if response.status_code == 200:
        results = response.json()["results"]
    else:
        print(f"Error querying {api_url}: {response.text}")
        results = []
    
    return results, elapsed_time

def check_relevance(song_title, query):
    """Use OpenAI API to check whether a song is relevant to the query."""
    prompt = (f"Determine if the song '{song_title}' is relevant to the topic '{query}'.\n"
              "Respond with only 'Relevant' or 'Not Relevant'.")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = response["choices"][0]["message"]["content"].strip()
    return result == "Relevant"

def evaluate_relevance(results):
    """Check relevance for a list of song results."""
    relevant_songs = [res["title"] for res in results[:K] if check_relevance(res["title"], QUERY_TEXT)]
    return relevant_songs

def precision_at_k(relevant_items, retrieved_items, k):
    retrieved_set = set(retrieved_items[:k])
    relevant_set = set(relevant_items)
    return len(retrieved_set & relevant_set) / k

def recall_at_k(relevant_items, retrieved_items, k):
    retrieved_set = set(retrieved_items[:k])
    relevant_set = set(relevant_items)
    return len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0

def ndcg_at_k(relevant_items, retrieved_items, k):
    relevance_scores = np.array([1 if item in relevant_items else 0 for item in retrieved_items[:k]])
    ideal_relevance = sorted(relevance_scores, reverse=True)
    return ndcg_score([ideal_relevance], [relevance_scores]) if len(ideal_relevance) > 0 else 0

# Query both models
gemini_results, gemini_time = query_model(GEMINI_API_URL, QUERY_TEXT)
huggingface_results, hf_time = query_model(HUGGINGFACE_API_URL, QUERY_TEXT)

# Extract relevant songs using OpenAI
gemini_relevant_songs = evaluate_relevance(gemini_results)
huggingface_relevant_songs = evaluate_relevance(huggingface_results)

# Extract retrieved song titles for evaluation
gemini_retrieved = [res["title"] for res in gemini_results[:K]]
huggingface_retrieved = [res["title"] for res in huggingface_results[:K]]

# Compute evaluation metrics for Gemini
gemini_eval = {
    "Precision@K": precision_at_k(gemini_relevant_songs, gemini_retrieved, K),
    "Recall@K": recall_at_k(gemini_relevant_songs, gemini_retrieved, K),
    "NDCG@K": ndcg_at_k(gemini_relevant_songs, gemini_retrieved, K),
}

# Compute evaluation metrics for HuggingFace
huggingface_eval = {
    "Precision@K": precision_at_k(huggingface_relevant_songs, huggingface_retrieved, K),
    "Recall@K": recall_at_k(huggingface_relevant_songs, huggingface_retrieved, K),
    "NDCG@K": ndcg_at_k(huggingface_relevant_songs, huggingface_retrieved, K),
}

# Prepare results
comparison_results = {
    "Query": QUERY_TEXT,
    "Response Time (s)": {
        "Gemini": gemini_time,
        "HuggingFace": hf_time
    },
    "Evaluation Metrics": {
        "Gemini": gemini_eval,
        "HuggingFace": huggingface_eval
    },
    "Relevant Songs (Gemini)": gemini_relevant_songs,
    "Relevant Songs (HuggingFace)": huggingface_relevant_songs,
    "Returned Content": {
        "Gemini": gemini_results[:K],
        "HuggingFace": huggingface_results[:K]
    }
}

# Save results to a JSON file
with open(OUTPUT_FILE, "w") as file:
    json.dump(comparison_results, file, indent=4)

print(f"Comparison results saved to {OUTPUT_FILE}")
print("Gemini Retrieved Songs:", gemini_retrieved)
print("HuggingFace Retrieved Songs:", huggingface_retrieved)
