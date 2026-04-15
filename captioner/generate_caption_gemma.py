import base64
import io
import json
import re
import os
import ollama
import psutil
import torch
from outlines import generator, models
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, Gemma4Processor


# GPU memory
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"GPU VRAM total: {props.total_memory / 1024**3:.1f} GB")
    print(f"GPU VRAM free:  {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
else:
    print("CUDA not available")



# CPU RAM
ram = psutil.virtual_memory()
print(f"CPU RAM total: {ram.total / 1024**3:.1f} GB")
print(f"CPU RAM free:  {ram.available / 1024**3:.1f} GB")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ── 1. Define your Pydantic caption schema ────────────────────────────────────
class ImageCaption(BaseModel):
    description: str = Field(
        description="A description of the main content of the image as well as ALL dialogue and general descriptions of the character appearances."
    )

class ChunkCaption(BaseModel):
    description: str = Field(
        description="A description of the main action and progression of the consecutive comic panels as well as ALL dialogue and descriptions of the character appearances, setting, and background."
    )
    summary: str = Field(
        description="A concise summary of the overall action taking place across the consecutive comic panels."
    )

# ── 2. Load model and processor ───────────────────────────────────────────────
MODEL_ID = "google/gemma-4-E2B-it"
#model_id = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     llm_int8_enable_fp32_cpu_offload=True)

# processor: Gemma4Processor = AutoProcessor.from_pretrained(MODEL_ID)

# hf_model = AutoModelForImageTextToText.from_pretrained(
#     MODEL_ID,
#     dtype=torch.bfloat16,
#     device_map="auto",
#     #attn_implementation="sdpa",
#     quantization_config=bnb_config,
#     offload_buffers=True,
#     low_cpu_mem_usage=True,
# ).eval()

# # Wrap with Outlines — pass processor (not tokenizer) for multimodal models
# # This enables hard-constrained JSON decoding against the Pydantic schema
# outlines_model = models.from_transformers(hf_model, processor)

# # Build the JSON generator, locked to the ImageCaption schema
# generator_model = generator.Generator(outlines_model, ImageCaption)

# # ── 3. Build the system prompt from the schema field descriptions ─────────────
# # Outlines enforces structure, but the model still needs to read
# # what values to produce — so describe the schema in the prompt too.
# schema_description = "\n".join(
#     f"- {name}: {field.description}"
#     for name, field in ImageCaption.model_fields.items()
#     if field.description
# )

# SYSTEM_PROMPT = f"""You are an image captioning assistant for a RAG pipeline.
# Analyze the image and return a JSON object with the following fields:
# {schema_description}
# Return only valid JSON. Do not include any explanation or extra text."""

# ── 4. Caption a single image ─────────────────────────────────────────────────
# def caption_image(image_path: str) -> ImageCaption:
#     image = Image.open(image_path).convert("RGB")

#     messages = [
#         {
#             "role": "system",
#             "content": SYSTEM_PROMPT,
#         },
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image",  "image": image},
#                 {"type": "text",   "text": "Caption this image."},
#             ],
#         },
#     ]

#     # Build the prompt string (Outlines needs a plain string, not token tensors)
#     prompt = processor.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=False,
#         token_budget=280,         # lower budget is recommended for captioning
#     )

#     # Generate — Outlines constrains output to valid ImageCaption JSON
#     result: ImageCaption = generator_model(prompt, max_tokens=512)
#     return result

def parse_caption(raw: str, schema: Union[ImageCaption, ChunkCaption]) -> ChunkCaption:
    """
    Robustly extract and validate JSON from model output.
    Handles markdown fences and leading/trailing text.
    """
    # Strip markdown code fences if present
    # e.g. ```json { ... } ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1)

    # If no fences, try to extract the first {...} block
    else:
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            raw = brace_match.group(0)

    try:
        data = json.loads(raw)
        return schema.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Could not parse model output as {schema.__class__.__name__}.\nRaw output:\n{raw}\nError: {e}")

def caption_image_ollama(image_path: Path) -> ImageCaption:
    schema = json.dumps(ImageCaption.model_json_schema(), indent=2)

    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

    response = ollama.chat(
        model="gemma4:e2b",
        messages=[{
                "role": "system",
                "content": (
                    f"You are a comic panel captioning assistant that understands text bubble dialogue and comic art. "
                    f"Always respond with valid JSON matching this schema exactly:\n{schema}\n"
                    f"Return only the JSON object. Use single quotes unless it is for a JSON field name. No markdown, no explanation, no code fences."
                )
                },
                {
                "role": "user",
                "content": (
                    "Caption this comic panel. Describe subjects, setting, "
                    "and all dialogue."
                ),
                "images": [image_b64]
            }],
        #format=schema,  # enforces Pydantic schema
        options={"temperature": 0,
                 "num_predict": 2048,
                 "num_ctx": 8192}                # deterministic output
    )

    print(response.get("prompt_eval_count"))  # tokens consumed by the input
    print(response.get("eval_count"))         # tokens generated in the response

    raw = response["message"]["content"]

    return parse_caption(raw, ImageCaption)

def caption_image_chunk_ollama(image_paths: list[Path]) -> ChunkCaption:
    schema = json.dumps(ChunkCaption.model_json_schema(), indent=2)

    img_64s = []

    for image_path in image_paths:
        with Image.open(image_path) as img:
            print(image_path.name)
            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="JPEG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            img_64s.append(image_b64)

    response = ollama.chat(
        model="gemma4:e2b",
        messages=[{
                "role": "system",
                "content": (
                    f"You are a comic panel captioning assistant that understands text bubble dialogue and comic art. "
                    f"Always respond with valid JSON matching this schema exactly:\n{schema}\n"
                    f"Return only the JSON object. Use single quotes unless it is for a JSON field name. No markdown, no explanation, no code fences."
                )
                },
                {
                "role": "user",
                "content": (
                    "Caption this consecutive series of comic panels. Describe subjects, setting, "
                    "and all dialogue as well as the general summary of action taking place across the panels."
                ),
                "images": img_64s
            }],
        #format=schema,  # enforces Pydantic schema
        options={"temperature": 0,
                 "num_predict": 2048,
                 "num_ctx": 8192}                # deterministic output
    )

    print(response.get("prompt_eval_count"))  # tokens consumed by the input
    print(response.get("eval_count"))         # tokens generated in the response

    raw = response["message"]["content"]

    return parse_caption(raw, ChunkCaption)


# ── 5. Batch-caption a folder ─────────────────────────────────────────────────
def caption_folder(image_dir: str) -> dict[str, ImageCaption]:
    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in extensions]

    captions: dict[str, ImageCaption] = {}
    for i, path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Captioning: {path.name}")
        try:
            captions[path.name] = caption_image_ollama(path)
        except Exception as e:
            print(f"  Warning: failed on {path.name}: {e}")

    return captions


# ── 6. Example usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = caption_image_ollama(Path(os.environ["PANEL_DIR"]) / "9_5.JPG")
    print(cap.description)
    # cap = caption_image_ollama(Path(os.environ["PANEL_DIR"]) / "4_3.JPG")
    # print(cap.description)
    # cap = caption_image_ollama(Path(os.environ["PANEL_DIR"]) / "4_4.JPG")
    # print(cap.description)

    cap = caption_image_chunk_ollama([Path(os.environ["PANEL_DIR"]) / "9_5.JPG",
                                      Path(os.environ["PANEL_DIR"]) / "9_6.JPG",
                                      Path(os.environ["PANEL_DIR"]) / "10_0.JPG"])

    # Fully typed Pydantic object
    print(cap.description)
    print(cap.summary)


    a=5