# -----------------------
# Indexing function (data_list uses GT text)
# -----------------------
import json
from pathlib import Path
import re
from typing import List

from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient

from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
from sentence_transformers import SentenceTransformer

from PIL import Image

from utils.util import ensure_dir

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
CHROMA_DB_PATH = Path("./e5_caption_chroma_image_db")

class QwenImageEmbedder(EmbeddingFunction):

    def __init__(self):
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True).eval().to(DEVICE)
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents with Qwen2.5 VL model
        embeddings = []
        for doc in input:
            emb = self.embed(doc)
            embeddings.append(emb)
        return embeddings
    
    def embed(self, input: str) -> List[float]:
        #print("Processing image for embedding:", input)
        inputs = self.processor.image_processor(images=Image.open(Path(input)).convert("RGB"), return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(DEVICE)
        grid_thw = inputs["image_grid_thw"].to(DEVICE)
        self.model.visual.to(pixel_values.device)
        with torch.no_grad():
            vision_output = self.model.visual(pixel_values, grid_thw)
            emb = vision_output.squeeze(0).mean(dim=0).cpu().numpy().tolist()

        return emb
    
class QwenTextEmbedder(EmbeddingFunction):

    def __init__(self):
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True).eval().to(DEVICE)
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents with Qwen2.5 VL model
        embeddings = []
        for doc in input:
            emb = self.embed(doc)
            embeddings.append(emb)
        return embeddings
    
    def embed(self, input: str) -> List[float]:
        #print("Processing image for embedding:", input)
        inputs = self.processor.tokenizer(input, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        
        with torch.no_grad():
            text_output = self.model.model(**inputs, output_hidden_states=True)
            text_output = text_output.hidden_states[-1]
            emb = text_output.mean(dim=1).cpu().numpy().tolist()[0]

        return emb

class E5SmallTextEmbedder(EmbeddingFunction):
    """
    Custom Chroma embedder class using intfloat/e5-small model.
    """

    def __init__(self, device: str = None):
        """
        Initialize the E5-small embedder.

        Args:
            device: Optional, 'cuda' or 'cpu'. If None, automatically selects GPU if available.
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading E5-small model on {self.device}...")
        self.model = SentenceTransformer("intfloat/e5-small").to(self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embed a list of strings.

        Args:
            input: List of text strings.

        Returns:
            List of embeddings (one list of floats per string).
        """
        return [self.embed(doc) for doc in input]

    def embed(self, input: str) -> List[float]:
        """
        Embed a single string.

        Args:
            input: Text string.

        Returns:
            Embedding as a list of floats.
        """
        with torch.no_grad():
            emb = self.model.encode([input], convert_to_numpy=True, show_progress_bar=False)[0]
        return emb.tolist()


def make_image_db(image_dir: Path):
    ensure_dir(CHROMA_DB_PATH)
    client = PersistentClient(path=CHROMA_DB_PATH)

    # create/get image collection
    image_db = client.get_or_create_collection(name="panel_image", 
                    embedding_function=E5SmallTextEmbedder())
    
    for panel in tqdm(image_dir.iterdir(), desc="Indexing images"):
        if panel.is_file():
            try:
                image_db.add(
                    documents=[str(panel)],
                    ids=[panel.stem],
                    metadatas=[{
                        "panel_id": panel.stem,
                        "image_path": str(panel)
                    }]
                )
            except Exception as e: 
                print(f"Failed to add image {panel}: {e}")

    print(f"Image database created at {CHROMA_DB_PATH} with {image_db.count()} images.")

def make_panel_db(caption_dir: Path, image_dir: Path):
    ensure_dir(CHROMA_DB_PATH)
    client = PersistentClient(path=CHROMA_DB_PATH)

    # create/get image collection
    image_db = client.get_or_create_collection(name="panel_image", 
                    embedding_function=E5SmallTextEmbedder())
    
    for panel in tqdm(caption_dir.iterdir(), desc="Indexing images"):
        if panel.is_file():
            try:
                with open(panel, "r", encoding="utf-8") as f:
                    data = json.load(f)

            # Compute embeddings separately
                image_text = data.get("image", "")
                main_text = data.get("text", "")
                image_db.add(
                    documents=[image_text, main_text],
                    ids=[f"{panel.stem}_img", f"{panel.stem}_text"],
                    metadatas=[{
                            "panel_id": panel.stem,
                            "field": "image",
                            "source_file": panel.name,
                            "source_img": str(image_dir / f"{panel.stem}.jpg")
                        },
                        {
                            "panel_id": panel.stem,
                            "field": "text",
                            "source_file": panel.name,
                            "source_img": str(image_dir / f"{panel.stem}.jpg")
                        }
                    ]
                )
            except Exception as e: 
                print(f"Failed to add image {panel}: {e}")

    print(f"Image database created at {CHROMA_DB_PATH} with {image_db.count()} images.")

def make_refined_panel_db(refined_captions_path: Path, image_dir: Path):
    ensure_dir(CHROMA_DB_PATH)
    client = PersistentClient(path=CHROMA_DB_PATH)

    # create/get collection for refined captions
    panel_db = client.get_or_create_collection(name="refined_panel_captions", 
                    embedding_function=E5SmallTextEmbedder())
    
    # Load the refined captions JSON
    with open(refined_captions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    captions = data.get("refined_captions", [])
    
    for caption in tqdm(captions, desc="Indexing refined captions"):
        # Parse PAGE and PANEL numbers using regex
        match = re.match(r"\s*PAGE (\d+) PANEL (\d+): (.*)", caption, re.DOTALL)
        if match:
            page_num = int(match.group(1))
            panel_num = int(match.group(2))
            description = match.group(3).strip()
            
            # Construct ID and image path
            panel_id = f"page{page_num}_panel{panel_num}"
            image_path = image_dir / f"{panel_id}.jpg"  # Adjust extension if needed (e.g., .png)
            
            try:
                panel_db.add(
                    documents=[caption],  # Use the full caption string as the document
                    ids=[panel_id],
                    metadatas=[{
                        "panel_id": panel_id,
                        "page": page_num,
                        "panel": panel_num,
                        "field": "refined_caption",
                        "source_img": str(image_path)
                    }]
                )
            except Exception as e:
                print(f"Failed to add caption for {panel_id}: {e}")
        else:
            print(f"Could not parse caption: {caption[:50]}...")

    print(f"Refined captions database created at {CHROMA_DB_PATH} with {panel_db.count()} captions.")


if __name__ == "__main__":
    caption_dir = Path(r"C:\Users\BabyBunny\Documents\Data\test_for_captioning\panel_captions")
    image_dir = Path(r"C:\Users\BabyBunny\Documents\Data\test_for_captioning\images")
    #make_panel_db(caption_dir=caption_dir, image_dir=image_dir)
    make_refined_panel_db(refined_captions_path=Path(r"C:\Users\BabyBunny\Documents\Data\test_for_captioning\panel_captions_openai_dialogue_refined\refined_captions.json"), image_dir=image_dir)
