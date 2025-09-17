import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio_simple(file_path, timeout=20, phrase_time_limit=None):
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Listening for audio input...")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            wav_data = audio.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            logging.info(f"Audio recorded and saved to {file_path}")

    except Exception as e:
        logging.error(f"Error during audio recording: {e}")

audio_filepath = "patient_voice_test.mp3"
record_audio_simple(audio_filepath)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

client = Groq(api_key=GROQ_API_KEY)
stt_model = "whisper-large-v3"

with open(audio_filepath, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model=stt_model,
        file=audio_file,
        language="en"
    )

print("Transcription:", transcription.text)
