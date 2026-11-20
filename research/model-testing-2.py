import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model
# model = SentenceTransformer('dunzhang/stella_en_400M_v5', truswt_remote_code=True, config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False})

# Sentence Transformer model: https://huggingface.co/jxm/cde-small-v1
model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)

# Maximum number of tokens to be passed into the model before being truncated
model.max_seq_length = 512
model.tokenizer.padding_side = "right"

# Load lyrics from a CSV file
csv_file_path = 'data/Songs.csv'  # Change this to your CSV file path
lyrics_df = pd.read_csv(csv_file_path)

# Assume we have a query to check against the lyrics
query = "hello"

# Function to add end-of-sequence token
def add_eos(input_examples):
    eos_token = model.tokenizer.eos_token
    if eos_token is not None:
        input_examples = [input_example + eos_token for input_example in input_examples]
    return input_examples

# Embed the query once
query_embedding = model.encode(add_eos([query]), batch_size=1, normalize_embeddings=True)

# Loop through each lyric and calculate the similarity score
for index, row in lyrics_df.iterrows():
    lyric = row['Lyrics']
    artist = row['Artist']
    title = row['Title']
    
    # Embed the current lyric
    lyric_embedding = model.encode(add_eos([lyric]), batch_size=1, normalize_embeddings=True)
    
    # Calculate cosine similarity score
    score = cosine_similarity(query_embedding, lyric_embedding) * 100

    # Print the results
    print(f"\nArtist: {artist}, Title: {title}")
    print("Similarity Score:", score[0][0])
