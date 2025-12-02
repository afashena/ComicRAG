# -----------------------
# Retrieval pipeline
# -----------------------
from typing import Dict, List

from chromadb import Collection

# Retrieval sizes
N_TEXT = 5
N_IMAGE = 5
N_TOTAL = 10

def retrieve(question: str, collection: Collection, top_text=N_TEXT, top_image=N_IMAGE, max_total=N_TOTAL) -> List[Dict]:

    query_result = collection.query(
            query_texts=[question], # Chroma embeds this text
            n_results=3         # Returns top 3 results
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
    context_blocks = []
    for c in candidates:
        context_blocks.append(f"PanelID: {c['panel_id']}\nComic: {c.get('comic_title','')}\nPage: {c.get('page_num')}\nText: {c.get('panel_text')}\nImagePath: {c.get('image_path')}\n")

    context = "\n\n---\n\n".join(context_blocks)
    prompt = f"""You are an archivist for a comic panel database. 
    Answer the user's question using ONLY the following candidate panels (cite PanelID and comic title + page if you reference it).

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