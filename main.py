"""
multimodal_rag_simple.py
Dual-index multimodal RAG (no reranker). 
- Images indexed with OpenCLIP ViT-bigG/14
- Text indexed with text-embedding-3-large (OpenAI) or fallback to intfloat/e5-large-v2
- Dual retrieval and placeholder merge function
"""

import os
import base64
from typing import List, Dict, Any

import torch
from PIL import Image
#import numpy as np

# Chroma
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# Transformers / CLIP
from transformers import CLIPModel, CLIPProcessor

# Optional local text embedder (E5)
from transformers import AutoTokenizer, AutoModel

from utils.util import ensure_dir

# Optional OpenAI embeddings + LLM
try:
    from openai import OpenAI  # new OpenAI python client
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHROMA_DB_PATH = "./chroma_multimodal_db"
CLIP_MODEL = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"  # recommended image embedder (OpenCLIP bigG)
# Text embedding preference: prefer OpenAI text-embedding-3-large if OPENAI_API_KEY set; fallback to e5-large-v2
E5_MODEL = "intfloat/e5-large-v2"

# Retrieval sizes
N_TEXT = 5
N_IMAGE = 5
N_TOTAL = 10

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

# -----------------------
# Text embedder: OpenAI (preferred) or local E5 fallback
# -----------------------
USE_OPENAI_EMBED = False
OPENAI_CLIENT = None
if "OPENAI_API_KEY" in os.environ and OPENAI_AVAILABLE:
    # Use OpenAI embeddings
    USE_OPENAI_EMBED = True
    OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    print("Using OpenAI text-embedding-3-large for text embeddings (requires API key).")
else:
    print("OpenAI API key not available or OpenAI client not installed. Falling back to local E5 model.")
    # load E5 local model
    e5_tok = AutoTokenizer.from_pretrained(E5_MODEL)
    e5_model = AutoModel.from_pretrained(E5_MODEL).to(DEVICE)
    e5_model.eval()

def embed_text_openai(texts: List[str]) -> List[List[float]]:
    # uses OpenAI client to generate embeddings
    # returns list of vectors
    resp = OPENAI_CLIENT.embeddings.create(model="text-embedding-3-large", input=texts)
    return [r.embedding for r in resp.data]

def embed_text_e5(texts: List[str]) -> List[List[float]]:
    # batch encode via E5
    inputs = e5_tok(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = e5_model(**inputs, return_dict=True)
        # mean pooling (standard for E5)
        last_hidden = out.last_hidden_state  # (batch, seq, dim)
        mask = inputs["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * mask).sum(1)
        counts = mask.sum(1)
        pooled = summed / counts
        # normalize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy().tolist()

def embed_texts(texts: List[str]) -> List[List[float]]:
    if USE_OPENAI_EMBED:
        return embed_text_openai(texts)
    else:
        return embed_text_e5(texts)

# -----------------------
# Initialize Chroma (two collections)
# -----------------------
ensure_dir(CHROMA_DB_PATH)
client = PersistentClient(path=CHROMA_DB_PATH)
dummy_ef = embedding_functions.DefaultEmbeddingFunction()

# create/get two collections
text_col = client.get_or_create_collection(name="panel_text", embedding_function=dummy_ef)
image_col = client.get_or_create_collection(name="panel_image", embedding_function=dummy_ef)

# -----------------------
# Indexing function (data_list uses GT text)
# -----------------------
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

# -----------------------
# Placeholder merge function
# -----------------------
def merge_candidates(text_hits: List[Dict], image_hits: List[Dict], max_total: int = 10) -> List[Dict]:
    """
    Placeholder — simple interleaving or union; replace with your preferred merging strategy later.
    Currently: concatenate text hits then image hits, dedupe by panel_id, limit to max_total.
    """
    combined = []
    seen = set()
    for hit in (text_hits + image_hits):
        panel_id = hit.get("panel_id") or hit["panel_id"]
        if panel_id not in seen:
            combined.append(hit)
            seen.add(panel_id)
        if len(combined) >= max_total:
            break
    return combined

# -----------------------
# Retrieval pipeline
# -----------------------
def retrieve(question: str, top_text=N_TEXT, top_image=N_IMAGE, max_total=N_TOTAL) -> List[Dict]:
    # 1) embed with text embedder
    q_text_vec = embed_texts([question])[0]

    # query text collection
    text_res = text_col.query(query_embeddings=[q_text_vec], n_results=top_text)
    # text_res structure: ids/metadatas
    text_hits = []
    if text_res and "metadatas" in text_res and len(text_res["metadatas"])>0:
        for m in text_res["metadatas"][0]:
            text_hits.append(m)

    # 2) CLIP text embedding (visual query) for image retrieval
    visual_query = question  # simple: we can also synthesize a more "visual" query using LLM later
    q_clip_text_vec = embed_clip_text(visual_query)
    image_res = image_col.query(query_embeddings=[q_clip_text_vec], n_results=top_image)
    image_hits = []
    if image_res and "metadatas" in image_res and len(image_res["metadatas"])>0:
        for m in image_res["metadatas"][0]:
            image_hits.append(m)

    # 3) Merge (placeholder)
    merged = merge_candidates(text_hits, image_hits, max_total=max_total)
    return merged

# -----------------------
# Compose LLM prompt and query LLM (example using OpenAI Chat if available)
# -----------------------
def llm_answer(question: str, candidates: List[Dict], openai_client=None) -> str:
    context_blocks = []
    for c in candidates:
        context_blocks.append(f"PanelID: {c['panel_id']}\nComic: {c.get('comic_title','')}\nPage: {c.get('page_num')}\nText: {c.get('panel_text')}\nImagePath: {c.get('image_path')}\n")

    context = "\n\n---\n\n".join(context_blocks)
    prompt = f"""You are an archivist for a comic panel database. Answer the user's question using ONLY the following candidate panels (cite PanelID and comic title + page if you reference it).

QUESTION:
{question}

CONTEXT:
{context}

Give a concise answer and list which panel(s) you relied on.
"""

    if openai_client is not None and OPENAI_AVAILABLE:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0
        )
        return resp.choices[0].message.content
    else:
        # If OpenAI LLM not available, return prompt + context (you can paste to any LLM).
        return prompt

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Example data (replace with your real dataset)
    data = [
        {"panel_id":"p1","image_path":"./data/rooftop_batman_joker.jpg","panel_text":"Batman: We need to talk, Joker.","comic_title":"The Laughing Bat #45","page_num":1},
        {"panel_id":"p2","image_path":"./data/inside_lab_joker.jpg","panel_text":"Joker: Stopped? That sounds so final, Batsy!","comic_title":"The Laughing Bat #45","page_num":1},
        {"panel_id":"p3","image_path":"./data/alleyway_robin.jpg","panel_text":"Robin: The coast is clear!","comic_title":"The Boy Wonder #12","page_num":3},
    ]

    index_panels(data)

    q = "When did Batman say 'We need to talk' to Joker?"
    candidates = retrieve(q)
    # If you have OpenAI client configured:
    if USE_OPENAI_EMBED and OPENAI_CLIENT is not None:
        ans = llm_answer(q, candidates, openai_client=OPENAI_CLIENT)
    else:
        ans = llm_answer(q, candidates, openai_client=None)
    print("ANSWER / PROMPT SENT TO LLM:")
    print(ans)
