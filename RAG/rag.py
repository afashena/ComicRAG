# -----------------------
# Retrieval pipeline
# -----------------------
import os
from pathlib import Path
from typing import Dict, List

from chromadb import Collection, PersistentClient
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from dotenv import load_dotenv
from openai import OpenAI
import torch

from utils.util import ensure_dir, image_to_b64

from database.make_db import E5SmallTextEmbedder, QwenImageEmbedder, QwenTextEmbedder

CHROMA_DB_PATH = Path("./e5_caption_chroma_image_db")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Retrieval sizes
N_TEXT = 5
N_IMAGE = 20
N_TOTAL = 10

def retrieve(question: str, collection: Collection, top_text=N_TEXT, top_image=N_IMAGE, max_total=N_TOTAL) -> List[Dict]:

    query_result = collection.query(
            query_texts=[question], # Chroma embeds this text
            n_results=top_image,         # Returns top results
            where={"field": "image"}
        )
    
    image_hits = []
    if query_result and "metadatas" in query_result and len(query_result["metadatas"])>0:
        for m in query_result["metadatas"][0]:
            image_hits.append(m)

    # 3) Merge (placeholder)
    #merged = merge_candidates(text_hits, image_hits, max_total=max_total)
    return image_hits

# -----------------------
# Compose LLM prompt and query LLM (example using OpenAI Chat if available)
# -----------------------
def llm_answer(question: str, candidates: List[Dict], openai_client=None) -> str:

    prompt = f"""You are an archivist for a comic panel database. 
    Answer the user's question using ONLY the following candidate panels (cite PanelID if you reference it).

    QUESTION:
    {question}

    Give a concise answer and list which panel(s) you relied on. If you do not know the answer, say "I don't know." and describe what happens in the first image.
    """

    context_blocks = []
    context_blocks.append({"type": "text", "text": prompt})
    for c in candidates:
        #context_blocks.append(f"PanelID: {c['panel_id']}\nImagePath: {c.get('image_path')}\n")
        text_img_pair = []
        text_img_pair.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_b64(c['source_img'])}", "detail": "auto"}})
        text_img_pair.append({"type": "text", "text": f"PanelID: {c['panel_id']}"})
        context_blocks.extend(text_img_pair)

    #context = "\n\n---\n\n".join(context_blocks)

    if openai_client is not None :
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":context_blocks}],
            temperature=0.0
        )
        return resp.choices[0].message.content
    else:
        # If OpenAI LLM not available, return prompt + context (you can paste to any LLM).
        return prompt
    
if __name__ == "__main__":
                        
    load_dotenv()  # load environment variables from .env file
    #get collection
    ensure_dir(CHROMA_DB_PATH)
    client = PersistentClient(path=CHROMA_DB_PATH)

    # create/get image collection
    image_db = client.get_or_create_collection(name="panel_image", embedding_function=E5SmallTextEmbedder())

    user_query = input("Enter your question about the comic panels: ")

    candidates = retrieve(user_query, collection=image_db, top_text=N_TEXT, top_image=N_IMAGE, max_total=N_TOTAL)

    openai_client = OpenAI()

    answer = llm_answer(user_query, candidates, openai_client=openai_client)
    print("\n=== ANSWER ===")
    print(answer)