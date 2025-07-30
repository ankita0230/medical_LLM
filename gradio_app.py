import gradio as gr
from brain_of_the_doctor import get_diagnosis
from voice_of_the_patient import speech_to_text
from voice_of_the_doctor import text_to_speech_with_gtts
from datetime import datetime
import os

# Ensure folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("audio_outputs", exist_ok=True)

# 🧠 Chat memory for UI (not passed to LLM)
chat_history = []

# 🔍 Diagnosis handler
def diagnose_and_speak(symptom_text, image=None, use_voice=False):
    global chat_history

    if not symptom_text and image is None:
        return "❌ Please provide symptoms or upload an image.", None, chat_history

    image_path = None
    if image:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_path = f"uploads/uploaded_image_{timestamp}.jpg"
        image.save(image_path)

    # Append user message to UI history
    chat_history.append(("user", symptom_text))

    # 🧠 Only send current input, not full history
    response = get_diagnosis(symptom_text, image_path)

    # Append assistant reply
    chat_history.append(("assistant", response))

    audio_path = None
    if use_voice:
        audio_path = text_to_speech_with_gtts(response)

    return response, audio_path, chat_history

# 🎙️ Voice-to-text appending
def append_speech_to_text(audio_path, current_text):
    if audio_path is None:
        return current_text
    speech_text = speech_to_text(audio_path)
    return (current_text + " " + speech_text) if current_text else speech_text

# 🔄 Reset chat
def reset_chat():
    global chat_history
    chat_history = []
    return "", None, []

# 🧠 Gradio UI
with gr.Blocks(title="🩺 Medical LLM Chatbot with Memory") as app:
    gr.Markdown("# 🧠 Medical Diagnosis Chatbot\nSupports voice, image, and chat memory")

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
            reset_button = gr.Button("🔄 New Disease / Reset Chat")

        with gr.Column():
            diagnosis_output = gr.Textbox(label="📋 Diagnosis Result", lines=10)
            audio_output = gr.Audio(label="🔊 Diagnosis Audio", type="filepath")
            chat_display = gr.Chatbot(label="💬 Chat History")

    # Event bindings
    submit_button.click(
        fn=diagnose_and_speak,
        inputs=[text_input, image_input, voice_checkbox],
        outputs=[diagnosis_output, audio_output, chat_display],
        show_progress=True
    )

    reset_button.click(
        fn=reset_chat,
        inputs=[],
        outputs=[diagnosis_output, audio_output, chat_display]
    )

app.launch()

