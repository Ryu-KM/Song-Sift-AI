import math
from flask import Flask, render_template, request, jsonify, url_for
import requests
import spotipy
import os
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up Spotipy with client credentials
client_credentials_manager = SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/results', methods=['GET'])
def results():
    # Retrieve the query input from the URL parameters
    query = request.args.get('query_input')
    if not query:
        return "Query input is required.", 400

    try:
        # Call the FastAPI service with the query parameter
        response = requests.get(f"{FASTAPI_BASE_URL}/query_lyrics", params={"query": query})
        response.raise_for_status()
        data = response.json()

        if "results" in data and isinstance(data["results"], list):
            songs_list = data["results"]

            # Sort songs by similarity score (descending)
            songs_list = sorted(songs_list, key=lambda x: x.get("similarity_score", 0), reverse=True)

            # Pagination settings: 5 songs per page
            page = int(request.args.get("page", 1))
            per_page = int(request.args.get("per_page", 5))
            total_pages = math.ceil(len(songs_list) / per_page)
            start = (page - 1) * per_page
            end = start + per_page
            current_page_songs = songs_list[start:end]

            # For each song, add album art, Spotify link info, and a LLM Summary
            for song in current_page_songs:
                artist = song.get("artist", "")
                title = song.get("title", "")
                album_art, spotify_link = get_album_art_and_link(title, artist)
                song["album_art"] = album_art if album_art else url_for('static', filename='defaultAlbum.png')
                song["spotify_link"] = spotify_link
                song["lyrics"] = song.get("lyric_sections", [])

            return render_template(
                "results.html",
                songs=current_page_songs,
                page=page,
                total_pages=total_pages,
                query_input=query
            )
        else:
            return render_template("results.html", songs=[], page=1, total_pages=1, query_input=query)
    except requests.exceptions.RequestException as e:
        print(f"Error calling FastAPI service: {e}")
        return "Error communicating with the FastAPI service.", 500

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    data = request.json
    artist = data.get("artist")
    title = data.get("title")
    lyrics = data.get("lyrics")

    if not artist or not title or not lyrics:
        return jsonify({"error": "Artist, title, and lyrics are required."}), 400

    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/generate_summary", json={
            "artist": artist,
            "title": title,
            "lyrics": lyrics
        })
        response.raise_for_status()
        summary = response.json().get("summary", "Summary unavailable.")
        return jsonify({"summary": summary})
    except requests.exceptions.RequestException as e:
        print(f"Error calling FastAPI service: {e}")
        return jsonify({"error": "Error communicating with the FastAPI service."}), 500


@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio data provided'}), 400

    audio_file = request.files['audio_data']
    # This will save the recording as a .wav file in /uploads
    filepath = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(filepath)
    # TODO: Add validation for file type and handle potential file overwrites.
    return jsonify({'message': 'File uploaded successfully', 'filename': audio_file.filename})

@app.route('/handle_audio_file', methods=['POST'])
def handle_audio_file():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio_data']
    mode = request.form.get('mode', 'text')

    original_filepath = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(original_filepath)

    wav_filename = os.path.splitext(audio_file.filename)[0] + '.wav'
    wav_filepath = os.path.join(UPLOAD_FOLDER, wav_filename)

    try:
        with open(original_filepath, 'rb') as wav_file:
            files = {
                'file': (os.path.basename(original_filepath), wav_file, 'audio/wav')
            }

            endpoint = '/match_snippet' if mode == 'audio' else '/transcribe_audio'
            response = requests.post(f"{FASTAPI_BASE_URL}{endpoint}", files=files)
            response.raise_for_status()

        fastapi_response = response.json()
        print(f"FastAPI response: {fastapi_response}")
        if mode == 'text':
            transcription_result = fastapi_response
            return jsonify({'transcription': transcription_result.get('transcription', '')})
        else:
            return jsonify({
            'message': 'Audio forwarded successfully',
            'filename': audio_file.filename,
            'mode': mode,
            'fastapi_response': fastapi_response
        })

    except requests.exceptions.RequestException as e:
        print(f"Error forwarding to FastAPI: {e}")
        return jsonify({'error': 'Failed to forward audio to FastAPI.'}), 500

    finally:
        # Clean up
        for f in [original_filepath, wav_filepath]:
            if os.path.exists(f):
                os.remove(f)

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Page not found"}), 404

def get_album_art_and_link(song_name, artist_name):
    """
    Fetch album art URL and Spotify link for a given song and artist.
    """
    try:
        query = f"track:{song_name} artist:{artist_name}"
        results = sp.search(q=query, type='track', limit=1)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            album_art_url = track['album']['images'][0]['url']
            spotify_link = track['external_urls']['spotify']
            return album_art_url, spotify_link
    except Exception as e:
        print(f"Error fetching album art and link from Spotify: {e}")
    return None, None

if __name__ == '__main__':
    app.run(port=5000, debug=True)