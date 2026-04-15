import os
import random
import pandas as pd
from pathlib import Path
import shutil

import json
from pathlib import Path
from pydantic import BaseModel
from typing import Any

from label_data import ComicPanelCaption


# ── Skeleton values by type ───────────────────────────────────────────────────
# One entry per field, with a placeholder that signals "fill this in"
# For lists, one skeleton item so you know the expected shape

def build_skeleton(model: type[BaseModel]) -> dict:
    """
    Derives a skeleton JSON structure from a Pydantic model.
    - str fields       -> ""
    - int/float fields -> 0
    - bool fields      -> False
    - Optional fields  -> None
    - Enum fields      -> "" (shows the field exists but needs filling)
    - List[str] fields -> [""]
    - List[Model]      -> [skeleton of that model]
    - Nested models    -> skeleton of that model
    """
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    return _skeleton_from_schema(schema, defs)


def _skeleton_from_schema(schema: dict, defs: dict) -> Any:
    # Resolve $ref (e.g. nested models and enums)
    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        schema = defs[ref_name]

    # anyOf is how Pydantic represents Optional[X]
    # It will be [{"$ref": "..."}, {"type": "null"}] or similar
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        if not non_null:
            return None
        # If it's Optional, use None as the skeleton value
        has_null = any(s.get("type") == "null" for s in schema["anyOf"])
        if has_null:
            return None
        return _skeleton_from_schema(non_null[0], defs)

    schema_type = schema.get("type")

    # Enum — show empty string so labeler knows to pick a value
    if "enum" in schema:
        return ""

    if schema_type == "object":
        result = {}
        properties = schema.get("properties", {})
        for field_name, field_schema in properties.items():
            result[field_name] = _skeleton_from_schema(field_schema, defs)
        return result

    if schema_type == "array":
        items_schema = schema.get("items", {})
        return [_skeleton_from_schema(items_schema, defs)]

    # Primitives
    if schema_type == "string":
        return ""
    if schema_type in ("integer", "number"):
        return 0
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None

    # Fallback
    return None

SKELETON = build_skeleton(ComicPanelCaption)


def generate_skeleton_jsons(
    chunk_dir: str,
    overwrite: bool = False
) -> None:
    """
    For each image in image_dir, write a sibling JSON file with
    empty/placeholder values ready to be filled in by hand.

    Args:
        chunk_dir:  folder containing your comic panel chunks
        overwrite:  if False, skips images that already have a JSON file
                    (safe to re-run without clobbering completed labels)
    """
    chunk_dir = Path(chunk_dir)

    for chunk_folder in chunk_dir.iterdir():
        if not chunk_folder.is_dir():
            continue

        extensions = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".webp", ".bmp"}
        image_paths = sorted(
            [p for p in chunk_folder.iterdir() if p.suffix.lower() in extensions],
            key=lambda p: natural_sort_key(str(p))
        )

        if not image_paths:
            print(f"No images found in {chunk_folder}")
            return

        skipped = 0
        created = 0

        for image_path in image_paths:
            json_path = chunk_folder / (image_path.stem + ".json")

            if json_path.exists() and not overwrite:
                skipped += 1
                continue

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(SKELETON, f, indent=2)

            created += 1
            print(f"  Created: {json_path.name}")

    print(f"\nDone. {created} created, {skipped} skipped (already exist).")

def generate_txts(
    chunk_dir: str,
    overwrite: bool = False
) -> None:
    """
    For each image in image_dir, write a sibling txt file with
    empty/placeholder values ready to be filled in by hand.

    Args:
        chunk_dir:  folder containing your comic panel chunks
        overwrite:  if False, skips images that already have a txt file
                    (safe to re-run without clobbering completed labels)
    """
    chunk_dir = Path(chunk_dir)

    for chunk_folder in chunk_dir.iterdir():
        if not chunk_folder.is_dir():
            continue

        extensions = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".webp", ".bmp"}
        image_paths = sorted(
            [p for p in chunk_folder.iterdir() if p.suffix.lower() in extensions],
            key=lambda p: natural_sort_key(str(p))
        )

        if not image_paths:
            print(f"No images found in {chunk_folder}")
            return

        skipped = 0
        created = 0

        for image_path in image_paths:
            txt_path = chunk_folder / (image_path.stem + ".txt")

            if txt_path.exists() and not overwrite:
                skipped += 1
                continue

            with open(txt_path, "w") as f:
                pass  # create an empty txt file

            created += 1
            print(f"  Created: {txt_path.name}")

    print(f"\nDone. {created} created, {skipped} skipped (already exist).")


def natural_sort_key(path: str) -> list:
    import re
    parts = re.split(r'[_\-.]', Path(path).stem)
    result = []
    for part in parts:
        try:
            result.append(int(part))
        except ValueError:
            result.append(part)
    return result


def create_chunks_from_comics(base_dir: str, chunk_size: int, chunk_num: int, output_dir: str):
    """
    Create a CSV file with randomly selected chunks of consecutive comic panels and save copies of the images in chunk-specific folders.

    Args:
        base_dir (str): Path to the base directory containing folders of comic panels.
        chunk_size (int): Number of consecutive panels in each chunk.
        chunk_num (int): Number of chunks to generate.
        output_dir (str): Path to save the folders containing chunk images.
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

    all_folders = [folder for folder in base_path.iterdir() if folder.is_dir()]

    chunks = []

    for i in range(chunk_num):
        # Randomly select a folder
        folder = random.choice(all_folders)
        
        # Get all panel files in the folder and sort them by name
        panel_files = sorted(folder.glob("*.jpg"), key=lambda x: x.stem)

        # Ensure there are enough panels for the chunk size
        if len(panel_files) < chunk_size:
            continue

        # Randomly select a starting index for the chunk
        start_idx = random.randint(0, len(panel_files) - chunk_size)
        chunk = panel_files[start_idx:start_idx + chunk_size]

        # Create a folder for this chunk
        chunk_folder = output_path / f"chunk_{i+1}"
        chunk_folder.mkdir(parents=True, exist_ok=True)

        # Copy images to the chunk folder
        for img in chunk:
            shutil.copy(img, chunk_folder / img.name)

        # Add the chunk to the list
        # chunks.append({
        #     "image_paths": [str(p) for p in chunk],
        #     "chunk_folder": str(chunk_folder),
        #     "caption": ""  # Empty caption column
        # })

    # Save to CSV
    # df = pd.DataFrame(chunks)
    # df.to_csv(output_csv, index=False)

# Example usage
# create_chunks_from_comics("path/to/comics", chunk_size=3, chunk_num=10, output_csv="output.csv", output_dir="output_chunks")

if __name__ == "__main__":
    comic_dir = Path(r"E:\Data\raw_panel_images\raw_panel_images")
    output_dir = Path(r"C:\Users\BabyBunny\Documents\Data\finetuning_set")
    # create_chunks_from_comics(base_dir=comic_dir, 
    #                           chunk_size=1, chunk_num=2000, 
    #                           output_dir=output_dir)
    # generate_skeleton_jsons(
    #     chunk_dir=output_dir,
    #     overwrite=False  # won't clobber labels you've already filled in
    # )
    generate_txts(
        chunk_dir=output_dir,
        overwrite=False  # won't clobber labels you've already filled in
    )