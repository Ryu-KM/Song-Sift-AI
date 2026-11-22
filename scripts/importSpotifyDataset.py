import json
import os
import re
import sys
import logging
import kagglehub
import pandas as pd

# init logger
os.makedirs('log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='log/importSpotifyDataset.log',
    filemode='a',
    encoding='utf-8'
    )

# Usage: python scripts/importSpotifyDataset.py [number_of_songs]   (default 20000, "all" for every song)
DATASET = "notshrirang/spotify-million-song-dataset"
OUTPUT_PATH = 'data/spotify_songs_with_lyrics.json'
EXISTING_PATHS = ['data/t_songs_with_lyrics.json', 'data/pop_songs_with_lyrics.json']
DEFAULT_SONGS_TO_IMPORT = 20000
RANDOM_SEED = 42

def clean_lyrics(text):
    # Dataset lines end in "  \r\n"; match the single-newline format of the Genius files
    lines = [line.strip() for line in text.replace('\r\n', '\n').split('\n')]
    return re.sub('\n{2,}', '\n', "\n".join(lines)).strip("\n")

def song_key(artist, title):
    return (artist.strip().lower(), title.strip().lower())


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_SONGS_TO_IMPORT)

    path = kagglehub.dataset_download(DATASET)
    df = pd.read_csv(os.path.join(path, "spotify_millsongdata.csv"))
    logging.info(f"Dataset has {len(df)} songs.")

    # Id from the original row number so ids stay the same between runs
    df['id'] = "spotify_" + df.index.astype(str)
    df = df.dropna(subset=['artist', 'song', 'text'])
    df = df.drop_duplicates(subset=['artist', 'song'])

    # Skip songs already collected from Genius
    existing_keys = set()
    for existing_path in EXISTING_PATHS:
        if os.path.exists(existing_path):
            with open(existing_path, 'r') as f:
                existing_keys.update(song_key(s['artist'], s['title']) for s in json.load(f))
    df = df[[song_key(a, t) not in existing_keys for a, t in zip(df['artist'], df['song'])]]

    if arg != "all" and int(arg) < len(df):
        df = df.sample(n=int(arg), random_state=RANDOM_SEED)

    songs = [{
        'title': row.song.strip(),
        'artist': row.artist.strip(),
        'url': row.link,
        'id': row.id,
        'lyrics': clean_lyrics(row.text)
    } for row in df.itertuples()]

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(songs, f, indent=4)
    logging.info(f"Wrote {len(songs)} songs to {OUTPUT_PATH}.")
