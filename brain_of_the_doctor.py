from groq import Groq
import os
from dotenv import load_dotenv
from image_analyzer import analyze_image

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama3-8b-8192"

client = Groq(api_key=GROQ_API_KEY)

def get_diagnosis(symptoms, image_path=None):
    image_desc = ""
    if image_path:
        try:
            analysis = analyze_image(image_path)
            top_labels = ", ".join([f"{label} ({prob*100:.1f}%)" for label, prob in analysis])
            image_desc = f"\n\nImage analysis suggests: {top_labels}."
        except Exception as e:
            image_desc = f"\n\n(Note: Image analysis failed: {e})"

    prompt = f"""You are a helpful medical assistant.
Patient reported the following symptoms: {symptoms}.{image_desc}

Provide:
1. Likely illness
2. Short description
3. Common medicine (optional)
4. Should patient visit a doctor?

Only give medical guidance — no legal disclaimers."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful and concise medical assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error from LLM: {e}"
