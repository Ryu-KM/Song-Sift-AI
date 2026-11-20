# Audio to Text Transcription 
import os
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

def convert_audio_to_wav(input_path, wav_path):
    """
    Convert an audio file (e.g., .webm, .mp3) to .wav format.
    """
    try:
        # Automatically detect the input file format
        audio = AudioSegment.from_file(input_path)
        audio.export(wav_path, format="wav")
        # Check if the conversion was successful
        return wav_path
    except Exception as e:
        print(f"Error converting {input_path} to {wav_path}: {e}")
        return None
    
def transcribe_audio_to_text(audio_path):    
    r = sr.Recognizer()
    # Use the SpeechRecognition library to transcribe the audio
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio, language='en-US')
        print("Transcription: " + text)
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Speech Recognition service; {0}".format(e))    

def transcribe_audio_blob_to_text(audio_blob):    
    # Step 1: Load WAV audio blob directly
    try:
        wav_io = BytesIO(audio_blob)
        wav_io.seek(0)  # Ensure cursor is at the beginning
    except Exception as e:
        print(f"Failed to load WAV audio blob: {e}")
        return ""

    # Step 2: Transcribe with SpeechRecognition
    r = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio, language='en-US')
        print("Transcription: " + text)
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Speech Recognition service; {e}")
        return ""

if __name__ == '__main__':
    wavfile = "recordings/recording.wav"
    transcribed_text = transcribe_audio_to_text(wavfile)  # Transcribe the audio file