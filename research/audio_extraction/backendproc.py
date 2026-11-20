import os
import sqlite3
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
import tempfile
import io

def create_database(db_name):
    # Create a database to store the features of the mel spectrograms
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS features
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  feature BLOB)''')
    conn.commit()
    conn.close()
    print(f"Database {db_name} created.")

def process_audio_file(file_path, db_name):
    try:
        # Check if the file exists in the database
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM features WHERE filename = ?", (file_path,))
        count = c.fetchone()[0]
        conn.close()
        if count > 0:
            print(f"File {file_path} already exists in the database. Skipping.")
            return

        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        if len(y) == 0:
            print("Error: Audio signal is empty.")
            return

        # Compute MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Use 13 coefficients (default)
        print(f"MFCC shape: {mfcc.shape}, dtype: {mfcc.dtype}")

        # Save the MFCCs to a binary blob in memory
        temp_buffer = io.BytesIO()
        np.save(temp_buffer, mfcc)
        temp_buffer.seek(0)  # Reset buffer pointer to the beginning
        blob_data = temp_buffer.read()

        # Store the MFCCs in the database
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute("INSERT INTO features (filename, feature) VALUES (?, ?)", (file_path, blob_data))
        conn.commit()
        conn.close()
        print(f"Stored MFCC features in database: {db_name}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# I need a function to take a snippet of audio and generate a mel spectrogram from it, and then compare it to all the mel spectrograms in the database to find the most similar one (the section could  be from any portion of the audio file)
def process_snippet(file_path, db_name):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        if len(y) == 0:
            print("Error: Audio signal is empty.")
            return

        # Compute MFCCs for the snippet
        mfcc_snippet = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        print(f"Snippet MFCC shape: {mfcc_snippet.shape}, dtype: {mfcc_snippet.dtype}")

        # Compare with database features
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute("SELECT id, filename, feature FROM features")
        rows = c.fetchall()

        best_match_id = None
        best_match_score = float('inf')
        best_match_filename = None

        for row in rows:
            db_id, filename, blob_data = row
            db_mfcc = np.load(io.BytesIO(blob_data))
            print(f"Comparing with {filename}, MFCC shape: {db_mfcc.shape}, dtype: {db_mfcc.dtype}")

            # Ensure the database MFCC is at least as large as the snippet
            if db_mfcc.shape[1] >= mfcc_snippet.shape[1]:
                snippet_length = mfcc_snippet.shape[1]
                num_windows = db_mfcc.shape[1] - snippet_length + 1

                for start_idx in range(num_windows):
                    # Extract a subsection of the database MFCC
                    db_mfcc_subsection = db_mfcc[:, start_idx:start_idx + snippet_length]

                    # Calculate similarity (Euclidean distance)
                    distance = np.linalg.norm(mfcc_snippet - db_mfcc_subsection)

                    if distance < best_match_score:
                        best_match_score = distance
                        best_match_id = db_id
                        best_match_filename = filename
            else:
                print(f"Skipping {filename} due to size mismatch.")

        conn.close()
        print(f"Best match ID: {best_match_id}, Filename: {best_match_filename}, Score: {best_match_score}")

    except Exception as e:
        print(f"Error processing snippet from file {file_path}: {e}")

# Example usage
def main():
    db_name = "audio_features.db"
    create_database(db_name)

    # Process an audio file and store its features in the database, get the first three songs in the song folder
    audio_folder = "research/audio_extraction/songs"  # Replace with your audio folder path
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.mp3')]
    for audio_file in audio_files:
        print(f"Processing file: {audio_file}")
        process_audio_file(audio_file, db_name)

    # Process a snippet of audio and compare it to the database
    snippet_file_path = "research/audio_extraction/BillieLunch.mp3"  # Replace with your snippet file path
    process_snippet(snippet_file_path, db_name)

main()


