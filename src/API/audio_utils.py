# audio_utils.py
import sqlite3
import numpy as np
import librosa
import io
from scipy.spatial.distance import cosine

def process_snippet(file_path, db_name):
    y, sr = librosa.load(file_path, sr=None)
    if len(y) == 0:
        raise ValueError("Audio signal is empty.")

    mfcc_snippet = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("SELECT id, filename, feature FROM features")
    rows = c.fetchall()

    best_match_id = None
    best_match_score = float('inf')
    best_match_filename = None

    for db_id, filename, blob_data in rows:
        db_mfcc = np.load(io.BytesIO(blob_data))
        if db_mfcc.shape[1] >= mfcc_snippet.shape[1]:
            snippet_length = mfcc_snippet.shape[1]
            num_windows = db_mfcc.shape[1] - snippet_length + 1
            for start_idx in range(num_windows):
                db_mfcc_subsection = db_mfcc[:, start_idx:start_idx + snippet_length]
                distance = np.linalg.norm(mfcc_snippet - db_mfcc_subsection)
                if distance < best_match_score:
                    best_match_score = distance
                    best_match_id = db_id
                    best_match_filename = filename
    conn.close()

    return {
        "best_match_id": int(best_match_id) if best_match_id is not None else None,
        "filename": best_match_filename,
        "distance": float(best_match_score)
    }

def process_chroma_snippet(file_path, db_name):
    y, sr = librosa.load(file_path, sr=None)
    if len(y) == 0:
        raise ValueError("Audio signal is empty.")

    # Extract Chroma features from the snippet (small portion of the song)
    chroma_snippet = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # Normalize and downsample for better performance
    chroma_snippet = librosa.util.normalize(chroma_snippet[:, ::5])  # Downsample snippet to reduce size

    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("SELECT id, filename, chroma FROM features")
    rows = c.fetchall()

    best_match_id = None
    best_match_score = float('inf')
    best_match_filename = None

    for db_id, filename, blob_data in rows:
        print(f"Processing {filename}...")

        # Load chroma from database and normalize
        db_chroma = np.load(io.BytesIO(blob_data))
        db_chroma = librosa.util.normalize(db_chroma[:, ::5])  # Downsample database chroma

        # Ensure both snippets have the same length by padding or trimming
        if chroma_snippet.shape[1] > db_chroma.shape[1]:
            continue  # Skip if snippet is larger than database feature (impossible to match)

        # Sliding window: slide a window of the snippet's length across the database chroma
        num_windows = db_chroma.shape[1] - chroma_snippet.shape[1] + 1  # Number of windows

        for start_idx in range(num_windows):
            db_window = db_chroma[:, start_idx:start_idx + chroma_snippet.shape[1]]

            # Compute cosine distance between the snippet and this window of the database chroma
            distance = cosine(chroma_snippet.flatten(), db_window.flatten())

            # Update best match if we found a better one
            if distance < best_match_score:
                best_match_score = distance
                best_match_id = db_id
                best_match_filename = filename

    conn.close()

    return {
        "best_match_id": int(best_match_id) if best_match_id is not None else None,
        "filename": best_match_filename,
        "distance": float(best_match_score)
    }
