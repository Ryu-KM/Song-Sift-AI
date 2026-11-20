import os
import sqlite3
import librosa
import numpy as np
import io

DB_NAME = "chroma_features.db"
SONG_DIR = "research/audio_extraction/songs"

def create_database():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            chroma BLOB
        )
    ''')
    conn.commit()
    conn.close()

def extract_and_store_chroma(file_path):
    y, sr = librosa.load(file_path, sr=None)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    print(f"Chroma shape for {file_path}: {chroma.shape}")

    # Save as binary
    buffer = io.BytesIO()
    np.save(buffer, chroma)
    buffer.seek(0)

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO features (filename, chroma) VALUES (?, ?)", 
              (os.path.basename(file_path), buffer.read()))
    conn.commit()
    conn.close()

def process_all_songs():
    create_database()
    for fname in os.listdir(SONG_DIR):
        if fname.endswith(".mp3"):
            fpath = os.path.join(SONG_DIR, fname)
            try:
                extract_and_store_chroma(fpath)
            except Exception as e:
                print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    process_all_songs()
