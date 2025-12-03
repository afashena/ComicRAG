# -----------------------
# Indexing function (data_list uses GT text)
# -----------------------
from pathlib import Path
from typing import Any, Dict, List

from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

import torch
from PIL import Image

from util import ensure_dir

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
CHROMA_DB_PATH = Path("./qwen_chroma_image_db")
#CLIP_MODEL = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"  # recommended image embedder (OpenCLIP bigG)

# -----------------------
# Load CLIP (image encoder + text tower for visual query)
# -----------------------
# print("Loading CLIP image model:", CLIP_MODEL)
# clip = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
# clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

# def embed_image(path: str) -> List[float]:
#     img = Image.open(path).convert("RGB")
#     inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         feat = clip.get_image_features(**inputs)  # shape (1, D)
#     feat = feat / feat.norm(dim=-1, keepdim=True)
#     return feat.cpu().numpy()[0].tolist()

# def embed_clip_text(text: str) -> List[float]:
#     # Use CLIP text encoder to obtain vector in CLIP image-text space
#     inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
#     with torch.no_grad():
#         feat = clip.get_text_features(**inputs)
#     feat = feat / feat.norm(dim=-1, keepdim=True)
#     return feat.cpu().numpy()[0].tolist()

class QwenImageEmbedder(EmbeddingFunction):

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True).eval().to(DEVICE)
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents with Qwen2.5 VL model
        embeddings = []
        for doc in input:
            emb = self.embed(doc)
            print(emb)
            embeddings.append(emb)
        return embeddings
    
    def embed(self, input: Documents) -> Embeddings:
        inputs = self.processor(images=[Image.open(Path(input)).convert("RGB")], return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)

        return emb.cpu().numpy().tolist()

def make_image_db(image_dir: Path):
    ensure_dir(CHROMA_DB_PATH)
    client = PersistentClient(path=CHROMA_DB_PATH)

    # create/get image collection
    image_db = client.get_or_create_collection(name="panel_image", 
                    embedding_function=QwenImageEmbedder())
    
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


if __name__ == "__main__":
    image_dir = Path(r"C:\Users\BabyBunny\Documents\Data\raw_panel_images\raw_panel_images\0")
    make_image_db(image_dir=image_dir)
