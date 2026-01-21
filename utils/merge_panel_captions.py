import json
from pathlib import Path
from util import ensure_dir

def merge_panel_captions(caption_dir: Path):

    ensure_dir(caption_dir)
    merged_captions: dict[str] = {"summary": ""}

    output_fp = caption_dir / "merged_captions.json"
    
    for caption_file in caption_dir.iterdir():
        if caption_file.is_file() and caption_file.suffix == ".json":
            with open(caption_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                #panel_id = caption_file.stem
                merged_captions["summary"] += (f"\n{data['panel_description']}")
    
    with open(output_fp, "w", encoding="utf-8") as out_f:
        json.dump(merged_captions, out_f, ensure_ascii=False, indent=4)
    
    print(f"Merged captions saved to {output_fp}")

if __name__ == "__main__":
    caption_directory = Path(r"C:\Users\BabyBunny\Documents\Data\test_for_captioning\panel_captions_w_plot")
    merge_panel_captions(caption_directory)