# uv add transformers accelerate pillow qwen-vl-utils pydantic
# note: the package is qwen-vl-utils (hyphen), imported as qwen_vl_utils (underscore)

import json
import os
from pathlib import Path
import re
import bitsandbytes
import transformers
import torch
from dotenv import load_dotenv

from PIL import Image
from pydantic import BaseModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = Path(r"C:\Users\BabyBunny\Documents\Models\qwen2.5-vl-3b-instruct")


# ── 1. Load model and processor (call once, reuse across many captions) ───────

def load_qwen(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    use_4bit: bool = True,
) -> tuple:
    """
    Load Qwen2.5-VL model and processor.

    Args:
        model_id: HuggingFace model identifier.
        use_4bit: quantize to 4-bit for low VRAM GPUs (your GTX 1050).
                  Set False on Lambda Cloud where VRAM is not a constraint.

    Returns:
        (model, processor) tuple — pass both to caption_image().
    """
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        # Cap image token budget to keep VRAM manageable on small GPUs.
        # 256*28*28 = ~200k pixels minimum, 512*28*28 = ~400k pixels maximum.
        # Raise max_pixels on Lambda if you want more visual detail.
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            #llm_int8_enable_fp32_cpu_offload=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )

    model.eval()
    return model, processor


def enforce_no_additional_properties(schema: dict) -> dict:
        """
        Recursively enforce additionalProperties=false
        on all object schemas (OpenAI strict requirement)
        """
        if not isinstance(schema, dict):
            return schema

        schema_type = schema.get("type")

        if schema_type == "object":
            schema["additionalProperties"] = False

            for prop in schema.get("properties", {}).values():
                schema = enforce_no_additional_properties(prop)

            # Required for OpenAI: explicitly define required
            if "required" not in schema and "properties" in schema:
                schema["required"] = list(schema["properties"].keys())

        elif schema_type == "array":
            schema = enforce_no_additional_properties(schema.get("items"))

        # Handle anyOf / oneOf / allOf (rare but safe)
        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema:
                for subschema in schema[key]:
                    schema = enforce_no_additional_properties(subschema)

        if "$defs" in schema:
            for def_schema in schema["$defs"].values():
                schema = enforce_no_additional_properties(def_schema)

        return schema


# ── 2. Core inference function ────────────────────────────────────────────────

def caption_image(
    image_path: str,
    model,
    processor,
    system_prompt: str,
    user_prompt: str,
    schema: type[BaseModel] | None = None,
    max_new_tokens: int = 1024,
) -> BaseModel | str:
    """
    Run Qwen2.5-VL on a single image and return either a Pydantic object
    (if schema is provided) or a raw string.

    Args:
        image_path:     Path to the image file.
        model:          Loaded Qwen model from load_qwen().
        processor:      Loaded processor from load_qwen().
        system_prompt:  Instruction context — who the model is and what it should do.
        user_prompt:    The specific question or instruction for this image.
        schema:         Optional Pydantic model class. If provided, the system prompt
                        is extended with the JSON schema and the output is parsed and
                        validated. If None, returns the raw string response.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Validated Pydantic instance if schema is provided, otherwise a plain string.
    """
    image = Image.open(image_path).convert("RGB")

    # If a schema is provided, extend the system prompt with the schema
    # and instruct the model to return only JSON.
    # This mirrors the approach that worked for you with Ollama.
    if schema is not None:
        schema_parsed = enforce_no_additional_properties(schema.model_json_schema())
        schema_str = json.dumps(schema_parsed, indent=2)
        full_system_prompt = (
            f"{system_prompt}\n\n"
            f"Always respond with a valid JSON object matching this schema exactly:\n"
            f"{schema_str}\n"
            f"Return only the JSON object. No markdown, no explanation, no code fences."
        )
    else:
        full_system_prompt = system_prompt

    messages = [
        {
            "role": "system",
            "content": full_system_prompt,
        },
        {
            "role": "user",
            "content": [
                # Qwen requires the image dict to use "image" key with a PIL Image
                # or a file path string — both work via process_vision_info
                {"type": "image", "image": image},
                {"type": "text",  "text": user_prompt},
            ],
        },
    ]

    # apply_chat_template produces the formatted text string
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # process_vision_info extracts image tensors from the message structure
    # This is Qwen-specific and required — do not skip it
    image_inputs, video_inputs = process_vision_info(messages)

    # Wrap entire inference in torch.inference_mode for maximum speed
    # This disables gradient computation and enables optimizations
    with torch.inference_mode():
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy — deterministic, better for structured output
        )

        # Trim the prompt tokens from the output — decode only what was generated
        trimmed = generated_ids[0][input_len:]
        raw_output = processor.decode(trimmed, skip_special_tokens=True).strip()

    if schema is None:
        return raw_output

    return _parse_to_schema(raw_output, schema)


# ── 3. Robust JSON extraction and Pydantic validation ────────────────────────

def _parse_to_schema(raw: str, schema: type[BaseModel]) -> BaseModel:
    """
    Extract JSON from model output and validate against the Pydantic schema.
    Handles markdown fences and leading/trailing text gracefully.
    """
    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        # Find the outermost {...} block
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = brace_match.group(0) if brace_match else raw

    try:
        data = json.loads(candidate)
        return schema.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(
            f"Could not parse model output as {schema.__name__}.\n"
            f"Raw output was:\n{raw}\n"
            f"Error: {e}"
        )


# ── 4. Example usage ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from typing import List, Optional
    from enum import Enum
    from pydantic import Field

    load_dotenv()

    # Your comic panel schema
    class ComicPanelCaption(BaseModel):
        description: str = Field(
            description=("Without knowing or making up narrative elements which are not in the panel, describe the setting, characters, and events evident in this comic panel as if you were telling a prose narrative."))

    # Load once
    model, processor = load_qwen(use_4bit=True)

    SYSTEM = (
        "You are a comic panel captioning assistant. "
        "Without knowing or making up narrative elements which are not in the panel, "
        "describe the setting, characters, and events evident in this comic panel "
        "as if you were telling a prose narrative."
    )

    # Structured output — returns a ComicPanelCaption instance
    caption = caption_image(
        image_path=Path(os.environ["PANEL_DIR"]) / "10_0.JPG",
        model=model,
        processor=processor,
        system_prompt=SYSTEM,
        user_prompt="Caption this comic panel.",
        schema=ComicPanelCaption,
    )
    print(type(caption))           # <class 'ComicPanelCaption'>
    print(caption.model_dump_json(indent=2))