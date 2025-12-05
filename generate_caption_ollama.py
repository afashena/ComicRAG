import os
import json
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, ValidationError, Field
from ollama import chat
from PIL import Image

# -----------------------------
# 1. DEFINE THE JSON SCHEMA
# -----------------------------
# 1. Define the Pydantic structure
class ImageDescription(BaseModel):
    image: str = Field(description="A detailed description of the people and their mood, environment, scenery, background, and colors of objects.")
    text: str = Field(description="Description of the dialogue bubbles, text, and any onomatopoeia present in the image.")


# -----------------------------
# 2. SETTINGS
# -----------------------------
MODEL_NAME = "qwen2.5vl:3b"   # or "gemma3:latest"
IMAGE_DIR = Path(r"C:\Users\BabyBunny\Documents\Data\test_for_captioning\images")
OUT_DIR = Path(r"C:\Users\BabyBunny\Documents\Data\test_for_captioning\panel_captions") 
MAX_RETRIES = 3


# -----------------------------
# 3. ENSURE OUTPUT DIR EXISTS
# -----------------------------
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------
# 4. CAPTION FUNCTION
# -----------------------------
def caption_image(image_path: str) -> ImageDescription:
    schema = ImageDescription.model_json_schema()

    prompt = (
        "Describe the vintage comic book panel and output ONLY valid JSON that follows exactly "
        "this schema:\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Do not add any text outside the JSON. "
    )

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[INFO] Captioning {image_path} (Attempt {attempt})")

        response = chat(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_path],
            }],
            format=schema,        # ← THIS ENFORCES JSON FROM THE MODEL
            options={
                "temperature": 0, # deterministic
                "device": "cuda"
            }
        )

        text = response["message"]["content"].strip()

        try:
            # Validate and parse JSON → Pydantic object
            parsed = ImageDescription.model_validate_json(text)
            return parsed

        except ValidationError as e:
            print("[WARN] JSON validation failed:", e)
            print("Raw output was:", text)
            print("Retrying…")

    raise RuntimeError(f"Failed to get valid JSON after {MAX_RETRIES} attempts.")


# -----------------------------
# 5. MAIN LOOP
# -----------------------------
def main():
    images = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    for img in images:
        img_path = os.path.join(IMAGE_DIR, img)
        print(f"[INFO] Processing: {img}")

        try:
            caption = caption_image(img_path)

            # Save JSON
            out_path = os.path.join(OUT_DIR, f"{Path(img).stem}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(caption.model_dump_json(indent=2))

            print(f"[SUCCESS] Saved → {out_path}")

        except Exception as e:
            print(f"[ERROR] Could not caption {img}: {e}")


if __name__ == "__main__":
    main()
