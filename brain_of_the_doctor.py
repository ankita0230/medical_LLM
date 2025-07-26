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
            image_desc = f"Image analysis shows: {top_labels}."
        except Exception as e:
            image_desc = f"Image analysis failed: {e}"

    prompt_parts = []

    if symptoms:
        prompt_parts.append(f"The patient describes the following symptoms:\n{symptoms}")

    if image_path:
        prompt_parts.append(f"{image_desc}")

    final_prompt = "\n\n".join(prompt_parts) + """

Please analyze the above patient information.

If the image findings and the reported symptoms match, combine both for a better diagnosis. If they are unrelated, explain that they might refer to different conditions. Then:

1. List the most likely illness.
2. Short description.
3. Common medicine (if any).
4. Should the patient visit a doctor?
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful and careful medical assistant."},
                {"role": "user", "content": final_prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error from LLM: {e}"
