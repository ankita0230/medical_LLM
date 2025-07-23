import gradio as gr
from brain_of_the_doctor import get_diagnosis
from voice_of_the_patient import speech_to_text
from voice_of_the_doctor import text_to_speech_with_gtts
from datetime import datetime
import os

# Ensure upload and output folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("audio_outputs", exist_ok=True)

def diagnose_and_speak(symptom_text, image=None, use_voice=False):
    if not symptom_text and image is None:
        return "❌ Please provide symptoms or upload an image.", None

    image_path = None
    if image:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_path = f"uploads/uploaded_image_{timestamp}.jpg"
        image.save(image_path)

    response = get_diagnosis(symptom_text, image_path)

    audio_path = None
    if use_voice:
        audio_path = text_to_speech_with_gtts(response)

    return response, audio_path


def append_speech_to_text(audio_path, current_text):
    if audio_path is None:
        return current_text
    speech_text = speech_to_text(audio_path)
    if current_text:
        return current_text + " " + speech_text
    else:
        return speech_text


with gr.Blocks(title="Medical LLM Chatbot") as app:
    gr.Markdown("# 🩺 Medical Diagnosis Chatbot")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Speak your symptoms")
            audio_button = gr.Button("Convert Speech to Text 📝")
            text_input = gr.Textbox(lines=4, placeholder="Or type your symptoms here...", label="🖊️ Enter Symptoms")

            audio_button.click(
                fn=append_speech_to_text,
                inputs=[audio_input, text_input],
                outputs=text_input
            )

            image_input = gr.Image(type="pil", label="🖼️ Optional: Upload related image (e.g. rash, eye)")
            voice_checkbox = gr.Checkbox(label="🔊 Read out the diagnosis")
            submit_button = gr.Button("💊 Diagnose")

        with gr.Column():
            diagnosis_output = gr.Textbox(label="📋 Diagnosis Result", lines=10)
            audio_output = gr.Audio(label="🔊 Diagnosis Audio", type="filepath")

    submit_button.click(
        fn=diagnose_and_speak,
        inputs=[text_input, image_input, voice_checkbox],
        outputs=[diagnosis_output, audio_output],
        show_progress=True
    )

app.launch()
