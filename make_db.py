# -----------------------
# Indexing function (data_list uses GT text)
# -----------------------
from typing import Any, Dict, List

from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from transformers import CLIPModel, CLIPProcessor

import torch
from PIL import Image

from util import ensure_dir

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHROMA_DB_PATH = "./chroma_image_db"
CLIP_MODEL = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"  # recommended image embedder (OpenCLIP bigG)

# -----------------------
# Load CLIP (image encoder + text tower for visual query)
# -----------------------
print("Loading CLIP image model:", CLIP_MODEL)
clip = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

def embed_image(path: str) -> List[float]:
    img = Image.open(path).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feat = clip.get_image_features(**inputs)  # shape (1, D)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0].tolist()

def embed_clip_text(text: str) -> List[float]:
    # Use CLIP text encoder to obtain vector in CLIP image-text space
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        feat = clip.get_text_features(**inputs)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0].tolist()

class TextEmbedder(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents with CLIP
        embeddings = []
        for doc in input:
            emb = embed_clip_text(doc)
            embeddings.append(emb)
        return embeddings

def make_image_db():
    ensure_dir(CHROMA_DB_PATH)
    client = PersistentClient(path=CHROMA_DB_PATH)

    # create/get image collection
    image_db = client.get_or_create_collection(name="panel_image", 
                    embedding_function=OpenCLIPEmbeddingFunction(model_name=CLIP_MODEL, device=DEVICE),
                    data_loader=ImageLoader())
    
    for panel in CHROMA_DB_PATH.iterdir():
        if panel.is_file():
            try:
                image_db.add(
                    documents=[str(panel)],
                    ids=[panel.stem]
                )
            except Exception as e:
                print(f"Failed to add image {panel}: {e}")

    print(f"Image database created at {CHROMA_DB_PATH} with {len(image_db)} images.")



def index_panels(data_list: List[Dict[str, Any]]):
    """
    data_list entries: {
        "panel_id": str,
        "image_path": str,
        "panel_text": str,
        "comic_title": str,  # optional
        "page_num": int      # optional
    }
    """
    text_ids, text_embs, text_metas = [], [], []
    img_ids, img_embs, img_metas = [], [], []

    # Collect texts for batch embedding for efficiency
    texts = [d["panel_text"] for d in data_list]
    text_vectors = embed_texts(texts)

    for i, d in enumerate(data_list):
        pid = d["panel_id"]
        img_path = d["image_path"]
        meta = {
            "panel_id": pid,
            "comic_title": d.get("comic_title", ""),
            "page_num": d.get("page_num", None),
            "panel_text": d["panel_text"],
            "image_path": os.path.abspath(img_path)
        }

        # text index entry
        text_ids.append(pid + "_text")
        text_embs.append(text_vectors[i])
        text_metas.append(meta)

        # image embedding (per-item)
        try:
            img_vec = embed_image(img_path)
        except Exception as e:
            print(f"Failed to embed image {img_path}: {e}")
            continue

        img_ids.append(pid + "_image")
        img_embs.append(img_vec)
        img_metas.append(meta)

    if text_ids:
        text_col.add(ids=text_ids, embeddings=text_embs, metadatas=text_metas)
    if img_ids:
        image_col.add(ids=img_ids, embeddings=img_embs, metadatas=img_metas)

    print(f"Indexed {len(text_ids)} text items and {len(img_ids)} image items.")
