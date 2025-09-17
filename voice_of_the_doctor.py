import os
import elevenlabs
from elevenlabs.client import ElevenLabs
import subprocess
import platfrom
from gtts import gTTS





def text_to_speech_with_gtts_old(input_text, output_filepath):
    language = "en"
    audioobj = gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)
    
    
input_text = "Hello, I am the Doctor. How can I assist you today?"
#text_to_speech_with_gtts_old(input_text=input_text, output_filepath="elevenlabs_testing.mp3")



ELEVENLABS_API_KEY=os.environ.get("ELEVENLABS_API_KEY")

def text_to_speech_with_elevenlabs_old(input_text=input_text, output_filepath="gtts_testing.mp3"):
    client=Elevenlabs(api_key=ELEVENLABS_API_KEY)
    audio = client.text_to_speech(
        text=input_text,
        voice="Aria",
        output_format="mp3_base64",
        model="eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)
#text_to_speech_with_elevenlabs_old(input_text=input_text, output_filepath="elevenlabs_testing.mp3")
#
#
def text_to_speech_with_gtts(input_text, output_filepath):
    language = "en"
    audioobj = gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)
    os_name = platform.system()

try:

    if os_name == "Darwin":  # macOS
        subprocess.run(['afplay', output_filepath])
    elif os_name == "Windows":  # Windows
        subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
    elif os_name == "Linux":  # Linux
        subprocess.run(['aplay', output_filepath])  # Alternative: use 'mpg123' or 'ffplay'
    else:
        raise OSError("Unsupported operating system")
except Exception as e:
    print(f"An error occurred while trying to play the audio: {e}")
    
    
input_text = "Hello, I am the Doctor. How can I assist you today?,autoplay testing"
text_to_speech_with_gtts(input_text=input_text, output_filepath="elevenlabs_testing.mp3")
##

def text_to_speech_with_elevenlabs(input_text=input_text, output_filepath="gtts_testing_autoplay.mp3"):
    client=Elevenlabs(api_key=ELEVENLABS_API_KEY)
    audio = client.text_to_speech(
        text=input_text,
        voice="Aria",
        output_format="mp3_base64",
        model="eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)
    
    os_name = platform.system()

try:

    if os_name == "Darwin":  # macOS
        subprocess.run(['afplay', output_filepath])
    elif os_name == "Windows":  # Windows
        subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
    elif os_name == "Linux":  # Linux
        subprocess.run(['aplay', output_filepath])  # Alternative: use 'mpg123' or 'ffplay'
    else:
        raise OSError("Unsupported operating system")
except Exception as e:
    print(f"An error occurred while trying to play the audio: {e}")
    
#text_to_speech_with_gtts(input_text=input_text, output_filepath="elevenlabs_testing.mp3") 
