import requests
import json
import time
import re
import os
from bs4 import BeautifulSoup

GENIUS_API_URL = "https://api.genius.com"
ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN") 
SEARCH_TERM = "t"
NEW_SONGS_TO_COLLECT = 100
SONGS_PER_PAGE = 50

def search_songs(page):
    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}'
    }
    params = {
        'q': SEARCH_TERM,
        'page': page
    }
    response = requests.get(f"{GENIUS_API_URL}/search", headers=headers, params=params)
    return response.json()

def collect_new_songs(existing_ids):
    collected_songs = []
    current_page = 1

    while len(collected_songs) < NEW_SONGS_TO_COLLECT:
        print(f"Fetching page {current_page}...")
        data = search_songs(current_page)

        if 'response' in data and 'hits' in data['response']:
            hits = data['response']['hits']
            for hit in hits:
                song_info = hit['result']
                song_id = song_info['id']
                if song_id in existing_ids:
                    continue  # Skip if already collected

                collected_songs.append({
                    'title': song_info['title'],
                    'artist': song_info['primary_artist']['name'],
                    'url': song_info['url'],
                    'id': song_id
                })

                if len(collected_songs) >= NEW_SONGS_TO_COLLECT:
                    break

            current_page += 1
            time.sleep(1)
        else:
            print("No more songs found.")
            break

    return collected_songs

def get_song_details(song_url):
    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}'
    }
    response = requests.get(song_url, headers=headers)
    html = BeautifulSoup(response.text.replace('<br/>', '\n'), "html.parser")
    divs = html.find_all("div", class_=re.compile("^lyrics$|Lyrics__Container"))
    if not divs:
        print(f"Couldn't find the lyrics section. Song URL: {song_url}")
        return "Lyrics not found"
    
    lyrics = "\n".join([div.get_text() for div in divs])
    lyrics = re.sub(r'(\[.*?\])*', '', lyrics)
    lyrics = re.sub('\n{2}', '\n', lyrics)
    return lyrics.strip("\n")


def extract_lyrics(song_details):
    try:
        children = song_details['response']['song']['description']['dom']['children']
        lyrics = ""
        for child in children:
            if isinstance(child, dict) and 'children' in child:
                for sub_child in child['children']:
                    if isinstance(sub_child, str):
                        lyrics += sub_child
        return lyrics
    except KeyError:
        return "Lyrics not found"


if __name__ == "__main__":
    # Load existing songs
    if os.path.exists('data/t_songs_with_lyrics.json'):
        with open('data/t_songs_with_lyrics.json', 'r') as f:
            songs = json.load(f)
        print(f"There are currently {len(songs)} songs in t_songs_with_lyrics.json'.")
    else:
        songs = []

    existing_ids = {song['id'] for song in songs}

    # Collect new songs
    new_songs = collect_new_songs(existing_ids)

    # Get lyrics for new songs
    for index, song in enumerate(new_songs):
        song_url = song['url']
        song['lyrics'] = get_song_details(song_url)
        print(f"Processed {index + 1}/{len(new_songs)}: {song['title']} by {song['artist']}")
        time.sleep(1)

    # Add new songs to file 
    songs.extend(new_songs)
    with open('data/t_songs_with_lyrics.json', 'w') as f:
        json.dump(songs, f, indent=4)
    print(f"Appended {len(new_songs)} new songs. Total is now {len(songs)} songs.")