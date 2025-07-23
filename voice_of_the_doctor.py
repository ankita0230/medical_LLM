from gtts import gTTS
import os
import time

def text_to_speech_with_gtts(text, output_dir="audio_outputs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = int(time.time())
    audio_path = os.path.join(output_dir, f"response_{timestamp}.mp3")

    try:
        tts = gTTS(text=text, lang='en')
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        return f"❌ Text-to-speech error: {e}"
