from tqdm import tqdm

from utils.util import image_to_b64, natural_key
from captioner.prompts import describe_panel, system_prompt, prose_system_prompt, prose_caption

from collections import deque
import os
import json
from pathlib import Path
from typing import Optional, List
from openai import OpenAI
from pydantic import BaseModel, ValidationError, Field
from ollama import chat, Client
from PIL import Image
from sqlalchemy import text
print(os.getcwd())

# -----------------------------
# 1. DEFINE THE JSON SCHEMA
# -----------------------------
# 1. Define the Pydantic structure
class ImageDescription(BaseModel):
    panel_description: str = Field(description="A detailed description of the comic panel including people and their mood if present, setting, and colors of objects. Include all dialogue bubbles and text along and describe the speakers.")
    #text: str = Field(description="Description of the dialogue bubbles, text, and any onomatopoeia present in the image including who says what.")

class PanelSummary(BaseModel):
    page_number: int = Field(description="The page number of the corresponding comic panel.")
    panel_number: int = Field(description="The panel number of the corresponding comic panel.")
    panel_prose: str = Field(description="A prose summary of the comic panel description which captures the dialogue, characters, setting, and plot points gathered from the full context of the story.")

class ComicSummary(BaseModel):
    summary: list[PanelSummary] = Field(description="A list of the prose summaries of the comic panels in order.")

class Character(BaseModel):
    description: str = Field(description="A detailed description of the character's appearance, clothing, and any notable features.")
    dialogue: str = Field(description="The character's dialogue or speech or thoughts as it appears in the comic panel. If there is no dialogue for this character, set this to be an empty string.")

class CharacterDescription(BaseModel):
    name: Optional[str] = Field(description="The character's name if it can be strongly inferred from the comic panel, otherwise set to null.")
    apparel: str = Field(description="A detailed description of the character's clothing.")
    skin_color: str = Field(description="The character's skin color in one or two words.")
    hair: str = Field(description="The character's hair color and style in a few words.")
    expression: str = Field(description="The character's facial expression and mood, if present.")
    dialogue: str = Field(description="The character's dialogue or speech or thoughts as it appears in the comic panel. If there is no dialogue for this character, set this to be an empty string.")

class PanelSummary2(BaseModel):
    setting: str = Field(description="A detailed description of the setting and background of the comic panel.")
    characters: List[CharacterDescription] = Field(description="A list of unique characters present in the comic panel along with their descriptions and dialogues.")
    other_text: str = Field(description="Any additional text present in the comic panel that is not dialogue such as onomatopoeia, signs, or captions.")

# class ScreenPlayFormat(BaseModel):
#     asdf: str = Field(description="asdf")

# class BatchSummary(BaseModel):
#     panel_description: str = Field(description="A detailed summary of the sequential comic panels including the setting, characters, actions, dialogue, and plot points.")
#     panel_summary: str = Field(description="A concise 2-3 sentence summary of the comic panel in the context of the current scene.")

class ComicCaptioner():
    def __init__(self, vl_model_name: str, l_model_name: str, image_dir: Path, out_dir: Path, caption_history: int, max_retries: int = 3):
        self.vl_model_name = vl_model_name
        self.l_model_name = l_model_name
        self.client = Client()
        self.image_dir = image_dir
        self.max_retries = max_retries
        self.summary: str = ""
        self.prev_captions: deque[str] = deque(maxlen=caption_history)  # store last few captions
        self.out_dir = out_dir

    def format_previous_captions(self) -> str:
        return "\n".join(self.prev_captions)
    
    def enforce_no_additional_properties(self, schema: dict) -> dict:
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
                self.enforce_no_additional_properties(prop)

            # Required for OpenAI: explicitly define required
            if "required" not in schema and "properties" in schema:
                schema["required"] = list(schema["properties"].keys())

        elif schema_type == "array":
            self.enforce_no_additional_properties(schema.get("items"))

        # Handle anyOf / oneOf / allOf (rare but safe)
        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema:
                for subschema in schema[key]:
                    self.enforce_no_additional_properties(subschema)

        if "$defs" in schema:
            for def_schema in schema["$defs"].values():
                self.enforce_no_additional_properties(def_schema)

        return schema



    def prompt_openai_w_image(self, prompt: str, image_path: Path) -> PanelSummary2:

        #schema = ImageDescription.model_json_schema()
        schema = PanelSummary2.model_json_schema()
        schema = self.enforce_no_additional_properties(schema)

        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_to_b64(str(image_path))}"
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "panel_caption",
                    "schema": schema,
                    "strict": True   # 🔑 important
                }
            }
        )

        caption = response.choices[0].message.content

        try:
            # Validate and parse JSON → Pydantic object
            parsed = PanelSummary2.model_validate_json(caption)
            return parsed

        except ValidationError as e:
            print("[WARN] JSON validation failed:", e)
            print("Raw output was:", caption)
            print("Retrying…")



    def prompt_model_w_image(self, prompt: str, image_path: Path) -> ImageDescription:
        schema = ImageDescription.model_json_schema()

        for attempt in range(1, self.max_retries + 1):
            print(f"[INFO] Captioning {image_path} (Attempt {attempt})")

            # response = chat(
            #     model=self.model_name,
            #     messages=[{"role": "system", 
            #                "content": system_prompt},
            #                 {
            #                 "role": "user",
            #                 "content": prompt,
            #                 "images": [image_path],
            #                 }
            #                 ],
            #     format=schema,        # ← THIS ENFORCES JSON FROM THE MODEL
            #     options={
            #         "temperature": 0, # deterministic
            #         "device": "cuda"
            #     }
            # )

            generate_prompt = f"{system_prompt}\n{prompt}<image>"

            response = self.client.generate(
                            model=self.vl_model_name,
                            prompt=generate_prompt,
                            images=[image_path],
                            format=schema,  # JSON schema enforcement
                            options={
                                "temperature": 0,
                                "device": "cuda"
                            }
                        )


            text = response.response.strip()

            try:
                # Validate and parse JSON → Pydantic object
                parsed = ImageDescription.model_validate_json(text)
                return parsed

            except ValidationError as e:
                print("[WARN] JSON validation failed:", e)
                print("Raw output was:", text)
                print("Retrying…")

        raise RuntimeError(f"Failed to get valid JSON after {self.max_retries} attempts.")
    
    # def prompt_model(self, prompt: str) -> ComicSummary:
    #     schema = ComicSummary.model_json_schema()

    #     for attempt in range(1, self.max_retries + 1):
    #         print(f"[INFO] Updating Summary (Attempt {attempt})")

    #         response = chat(
    #             model=self.model_name,
    #             messages=[{"role": "system", 
    #                        "content": prose_system_prompt},
    #                         {
    #                         "role": "user",
    #                         "content": prompt,
    #                     }],
    #             format=schema,        # ← THIS ENFORCES JSON FROM THE MODEL
    #             options={
    #                 "temperature": 0, # deterministic
    #                 "device": "cuda"
    #             }
    #         )

    #         text = response["message"]["content"].strip()

    #         try:
    #             # Validate and parse JSON → Pydantic object
    #             parsed = ComicSummary.model_validate_json(text)
    #             return parsed

    #         except ValidationError as e:
    #             print("[WARN] JSON validation failed:", e)
    #             print("Raw output was:", text)
    #             print("Retrying…")

    #     raise RuntimeError(f"Failed to get valid JSON after {self.max_retries} attempts.")

    def get_panel_descriptions(self) -> str:

        panel_descriptions = []

        jsons = sorted(
                    [
                        f for f in os.listdir(self.out_dir)
                        if f.lower().endswith((".json"))
                    ],
                    key=natural_key
                )
        
        jsons = [self.out_dir / desc_path for desc_path in jsons]

        for desc_path in jsons:
            with open(desc_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                #desc = json_data["panel_description"]

                page_num = desc_path.stem.split("_")[0]
                panel_num = desc_path.stem.split("_")[1]
                panel_descriptions += [f"\nPAGE {page_num} PANEL {panel_num}: \n{json_data}"]

        return panel_descriptions

    def refine_caption(self) -> ComicSummary:

        client = OpenAI()

        schema = ComicSummary.model_json_schema()
        schema = self.enforce_no_additional_properties(schema)

        chunk_size = 3

        panel_descriptions = self.get_panel_descriptions()
        updated_descriptions = [None] * len(self.get_panel_descriptions())

        for idx in tqdm(range(len(panel_descriptions)), desc="Refining captions: "):

            batch = panel_descriptions[idx : idx + chunk_size]

            #prompt = f"{prose_system_prompt}\n{prose_caption.format(panel_descriptions=batch,)}"
            
            # response = self.client.generate(
            #     model=self.l_model_name,
            #     prompt=prompt,
            #     format=schema,
            #     options={"temperature": 0, "device": "cuda"}
            # )

            # result = response.response.strip()

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prose_system_prompt},
                    {"role": "user", "content": f"Refine this caption: {prose_caption.format(panel_descriptions=batch,)}"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ComicSummary",
                        "schema": schema,
                        "strict": True
                        }
                },
            )
            result = response.choices[0].message.content

            try:
                # Validate and parse JSON → Pydantic object
                parsed = ComicSummary.model_validate_json(result)
                for i, summary in enumerate(parsed.summary):
                    updated_descriptions[idx + i] = f"\nPAGE {summary.page_number} PANEL {summary.panel_number}: \n{summary.panel_prose}"

            except ValidationError as e:
                print("[WARN] JSON validation failed:", e)
                print("Raw output was:", result)
                print("Retrying…")

        # save updated descriptions as a single JSON
        out_path = self.out_dir / "refined_captions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"refined_captions": updated_descriptions}, f, indent=4)

    def caption_panel(self, image_path: Path) -> ImageDescription:
        prompt = describe_panel
        #caption = self.prompt_model_w_image(prompt, image_path)
        caption = self.prompt_openai_w_image(prompt, image_path)
        return caption

    def run(self):
        images = sorted(
                    [
                        f for f in os.listdir(self.image_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                    ],
                    key=natural_key
                )

        for idx, img in enumerate(images):
            img_path = self.image_dir / img
            print(f"[INFO] Processing: {img}")

            try:
                caption = self.caption_panel(image_path=img_path)

                # Save JSON
                out_path = self.out_dir / f"{Path(img).stem}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(caption.model_dump_json(indent=4))

                print(f"[SUCCESS] Saved → {out_path}")

            except Exception as e:
                print(f"[ERROR] Could not caption {img}: {e}")


if __name__ == "__main__":
    vl_model_name = "qwen2.5vl:3b"   # or "gemma3:latest"
    l_model_name = "mistral:7b-instruct-q4_K_M"
    image_dir = Path(r"C:\Users\BabyBunny\Documents\Data\test_for_captioning\images")
    out_dir = Path(r"C:\Users\BabyBunny\Documents\Data\test_for_captioning\panel_captions_openai_dialogue_refined") 
    max_retries = 3

    
    Path.mkdir(out_dir, exist_ok=True)

    captioner = ComicCaptioner(vl_model_name=vl_model_name,
                               l_model_name=l_model_name,
                               image_dir=image_dir,
                               out_dir=out_dir,
                               caption_history=3,
                               max_retries=max_retries)
    #captioner.run()
    captioner.refine_caption()
    print("Done captioning!")
