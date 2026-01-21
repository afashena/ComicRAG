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
from utils.util import image_to_b64, natural_key
from captioner.prompts import describe_panel_batch, describe_scene_summary, describe_panel, batch_sys_prompt, system_prompt, summary_system_prompt, describe_panel_w_history, prose_system_prompt, prose_caption

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

class PanelSummary2(BaseModel):
    setting: str = Field(description="A detailed description of the setting and background of the comic panel.")
    characters: List[Character] = Field(description="A list of unique characters present in the comic panel along with their descriptions and dialogues.")
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

        schema = ComicSummary.model_json_schema()

        chunk_size = 3

        panel_descriptions = self.get_panel_descriptions()

        updated_descriptions = panel_descriptions

        for idx in range(len(updated_descriptions)):

            if idx == 5:
                a = 5

            batch = updated_descriptions[idx : idx + chunk_size]

            prompt = f"{prose_system_prompt}\n{prose_caption.format(
                panel_descriptions=batch,)}"
            
            response = self.client.generate(
                model=self.l_model_name,
                prompt=prompt,
                format=schema,
                options={"temperature": 0, "device": "cuda"}
            )

            result = response.response.strip()

            try:
                # Validate and parse JSON → Pydantic object
                parsed = ComicSummary.model_validate_json(result)
                for i, summary in enumerate(parsed.summary):
                    updated_descriptions[idx + i] = f"\nPAGE {summary.page_number} PANEL {summary.panel_number}: \n{summary.panel_prose}"
                #return parsed

            except ValidationError as e:
                print("[WARN] JSON validation failed:", e)
                print("Raw output was:", result)
                print("Retrying…")

        return

    def caption_panel(self, image_path: Path) -> ImageDescription:
        prompt = describe_panel
        #caption = self.prompt_model_w_image(prompt, image_path)
        caption = self.prompt_openai_w_image(prompt, image_path)
        #self.prev_captions.append(caption.panel_summary)
        return caption

    # def update_scene_summary(self, caption: PanelSummary) -> None:
    #     prompt = describe_scene_summary.format(
    #         old_summary=self.summary,
    #         caption_i=caption.panel_description
    #     )
    #     updated_summary = self.prompt_model(prompt) 
    #     self.summary = updated_summary.summary

    def run_batch(self):
        images = sorted(
                    [
                        f for f in os.listdir(self.image_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                    ],
                    key=natural_key
                )
        
        image_paths = [self.image_dir / img_path for img_path in images]
        
        # caption = self.caption_panel(is_first=False, image_path=image_paths[0:3])
        # self.update_scene_summary(caption)

        batch_size = 3
        for idx in range(0, len(image_paths), batch_size):
            print(f"[INFO] Processing: {image_paths[idx:idx+batch_size]}")

            try:
                caption = self.caption_panel(is_first=(idx == 0), image_path=image_paths[idx : idx + batch_size])
                #self.update_scene_summary(caption)

                # Save JSON
                out_path = self.out_dir / f"{idx}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(caption.model_dump_json(indent=2))

                print(f"[SUCCESS] Saved → {out_path}")

            except Exception as e:
                print(f"[ERROR] Could not caption {image_paths[idx:idx+batch_size]}: {e}")

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
                #self.update_scene_summary(caption)

                # Save JSON
                out_path = self.out_dir / f"{Path(img).stem}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(caption.model_dump_json(indent=4))

                print(f"[SUCCESS] Saved → {out_path}")

            except Exception as e:
                print(f"[ERROR] Could not caption {img}: {e}")
# -----------------------------
# 2. SETTINGS
# -----------------------------



# -----------------------------
# 3. ENSURE OUTPUT DIR EXISTS
# -----------------------------



# -----------------------------
# 4. CAPTION FUNCTION
# -----------------------------
def caption_image(image_path: str) -> PanelSummary:
    schema = PanelSummary.model_json_schema()

    prompt = (
        "Describe the vintage comic book panel and output ONLY valid JSON that follows exactly "
        "this schema:\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Do not add any text outside the JSON. "
    )

    for attempt in range(1, max_retries + 1):
        print(f"[INFO] Captioning {image_path} (Attempt {attempt})")

        response = chat(
            model=model_name,
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
            parsed = PanelSummary.model_validate_json(text)
            return parsed

        except ValidationError as e:
            print("[WARN] JSON validation failed:", e)
            print("Raw output was:", text)
            print("Retrying…")

    raise RuntimeError(f"Failed to get valid JSON after {max_retries} attempts.")

def caption_image_w_context(image_path: str, all_captions: str) -> PanelSummary:
    schema = PanelSummary.model_json_schema()

    prompt = (
        f"""Describe the vintage comic book panel in the associated image and output ONLY valid JSON that follows exactly 
        this schema:
        f"{json.dumps(schema, indent=2)}
        
        Do not add any text outside the JSON. 
        
        This comic panel has the following context from previous panels. The following text does not describe the current panel. Do not repeat this text, but use it to inform your description of the current panel: 
        {all_captions}
        """
    )

    for attempt in range(1, max_retries + 1):
        print(f"[INFO] Captioning {image_path} (Attempt {attempt})")

        response = chat(
            model=model_name,
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
            parsed = PanelSummary.model_validate_json(text)
            return parsed

        except ValidationError as e:
            print("[WARN] JSON validation failed:", e)
            print("Raw output was:", text)
            print("Retrying…")

    raise RuntimeError(f"Failed to get valid JSON after {max_retries} attempts.")

def caption_images(image_paths: list[str]) -> ImageDescription:
    schema = ImageDescription.model_json_schema()

    prompt = (
        "Describe the vintage comic book panel and output ONLY valid JSON that follows exactly "
        "this schema:\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Do not add any text outside the JSON. "
    )

    for attempt in range(1, max_retries + 1):
        print(f"[INFO] Captioning {image_paths} (Attempt {attempt})")

        response = chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_path for image_path in image_paths],
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

    raise RuntimeError(f"Failed to get valid JSON after {max_retries} attempts.")

# -----------------------------
# 5. MAIN LOOP
# -----------------------------
def main():
    images = sorted(
    [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ],
    key=natural_key
)

    all_captions = ""

    for img in images:
        img_path = os.path.join(image_dir, img)
        print(f"[INFO] Processing: {img}")

        try:
            caption = caption_image_w_context(img_path, all_captions)
            all_captions += f" {caption.summary}"

            # Save JSON
            out_path = os.path.join(out_dir, f"{Path(img).stem}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(caption.model_dump_json(indent=2))

            print(f"[SUCCESS] Saved → {out_path}")

        except Exception as e:
            print(f"[ERROR] Could not caption {img}: {e}")

    # save all captions to a single file
    merged_output_fp = out_dir / "merged_captions.json"
    with open(merged_output_fp, "w", encoding="utf-8") as out_f:
        json.dump({"summary": all_captions.strip()}, out_f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    vl_model_name = "qwen2.5vl:3b"   # or "gemma3:latest"
    l_model_name = "mistral:7b-instruct-q4_K_M"
    image_dir = Path(r"C:\Users\BabyBunny\Documents\Data\test_for_captioning\images")
    out_dir = Path(r"C:\Users\BabyBunny\Documents\Data\test_for_captioning\panel_captions_openai_dialogue") 
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
    #main()
