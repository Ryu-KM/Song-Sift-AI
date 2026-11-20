# Sound-Sift-AI
AI Similar Content Extractor Drexel University senior project 2025 (continued)

## Build

### Prerequisites

#### Python

The AI Similar Content Extractor is a Python base project and you will need The Python version 3.12 specifically for the application to work properly. 

For Windows the installation can be through the Microsoft Store by simply searching for Python and locating the 3.12 version which should be the second option listed in the app store.

#### Build Tools

For this application to successfully install all the needed modules, that you will have to have to download some build tools in order to have access to everything required for the build.

First navigate to this [link](https://visualstudio.microsoft.com/downloads/?q=build+tools) 

This will take you to the download page for Visual Studio Tools, you will select the free download option and follow it's prompts untill you manage to get to the Workload screen where you can select which options you would like to install. You will only need to select one option which is the Desktop development with C++ under the Desktop & Mobile section. Once you have checked this option you may continue to install. This will take a minute.

#### Modules

The required modules are listed in the `requirements.txt` file. To install them, run the following command:

```bash
pip install -r requirements.txt
```

#### Virtual Enviroment

We will need to set up a virtual enviroment before running the code.

`mkdir venv.`
This command will wreate a new directory named venv, and it will hold the virtual environment files, keeping all dependencies isolated.

`cd .\venv\`
This command will move us into the directory we just made.

`python -m venv .`
This command sets up a virtual environment within the venv folder, and this will allow you to install and manage Python packages independently from the global Python installation. This should help to keep the project dependencies organized.

#### Adding items to ChromaDB

First run the script:
`research/add-to-chroma.py`
This script `research/add-to-chroma.py` is responsible for creating a Chroma database, which will hold vectorized embeddings of text data.
They will then be stored in `research/chromaFiles`.

Then you can run the script:
`research/query_embeddings.py`
There is a variable labeled `query_text` where you input the query you want to search for in the database, and you should recieve similar entries based on your query text.

## Running application

To run the application, there are two different options at the moment which is to either use Gemini or a locally sourced model from Hugging Face.

### To use Gemini

1. Install the required Python modules listed in the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

2. Generate an API key from [Google AI Studio](https://aistudio.google.com/app/apikey) and set the environment variable with this key. Also, generate a Spotify API Client_ID and Client_Secret from [Spotify Developer](https://developer.spotify.com/documentation/web-api) and set the environment variables.

    On Linux/macOS run:
    ```bash
    export GEMINI_API_KEY='your_gemini_api_key'
    export SPOTIPY_CLIENT_ID='your_client_id'
    export SPOTIPY_CLIENT_SECRET='your_client_secret'
    ```

    On Windows in a CMD as admin run:
    ```powershell
    setx GEMINI_API_KEY your_gemini_api_key
    setx SPOTIPY_CLIENT_ID your_client_id
    setx SPOTIPY_CLIENT_SECRET your_client_secret
    ```

3. In the root directory run `python research/gemini-add-to-chroma.py`

4. cd in to `src/api/` and run `uvicorn geminiMain:app --reload`

5. In a separate terminal cd in to `src/webservice/` and run `python flaskapp.py`

6. Visit the URL given after running the flask app and you should see the frontend for the application.

### To use a locally sourced model

1. Install the required Python modules listed in the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

2. Generate a Spotify API Client_ID and Client_Secret from [Spotify Developer](https://developer.spotify.com/documentation/web-api) and set the environment variables.

    On Linux/macOS run:
    ```bash
    export SPOTIPY_CLIENT_ID='your_client_id'
    export SPOTIPY_CLIENT_SECRET='your_client_secret'
    ```

    On Windows in a CMD as admin run:
    ```powershell
    setx SPOTIPY_CLIENT_ID your_client_id
    setx SPOTIPY_CLIENT_SECRET your_client_secret
    ```

2. In the root directory run `python research/add-to-chroma.py`

3. cd in to `src/api/` and run `uvicorn main:app --reload`

4. In a separate terminal cd in to `src/webservice/` and run `python flaskapp.py`

5. Visit the URL given after running the flask app and you should see the frontend for the application.

### To use the transcription feature
1. You must install `ffmpeg`. 
2. [Click here to install](https://ffmpeg.org/download.html), select the corresponding operating system (Linux, Windows, MacOS, etc.) and follow the documentation
