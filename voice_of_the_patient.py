import speech_recognition as sr

def speech_to_text(audio_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "❌ Could not understand the audio"
    except sr.RequestError as e:
        return f"❌ Could not request results; {e}"
