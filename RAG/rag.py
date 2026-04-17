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
from pydantic import BaseModel, Field, ValidationError
import torch
from FlagEmbedding import FlagReranker
from jina import Flow, Document, DocumentArray
from PIL import Image
import io

from utils.util import ensure_dir, image_to_b64

from vector_db.make_db import E5SmallTextEmbedder, QwenImageEmbedder, QwenTextEmbedder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.__version__)               # The CUDA version PyTorch was compiled against
print(torch.version.cuda)          # The CUDA version PyTorch is using at runtime
print(torch.cuda.device_count())        # 0 means no GPUs detected

# Retrieval sizes
N_TEXT = 5
N_IMAGE = 20
N_TOTAL = 10

class Answer(BaseModel):
    page_panel: str = Field(description="The string identifier for the panel that best answers the user's question in the format '{page_num}_{panel_num}', e.g., '1_3' for page 1 panel 3.")
    answer_text: str = Field(description="The answer to the user's question based on the content of the panel. This should be a concise summary that directly addresses the question, using information from the panel's image and caption. If the answer is not known, this should be 'I don't know.' followed by a brief description of what happens in the panel.")

def enforce_no_additional_properties(schema: dict) -> dict:
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
                enforce_no_additional_properties(prop)

            # Required for OpenAI: explicitly define required
            if "required" not in schema and "properties" in schema:
                schema["required"] = list(schema["properties"].keys())

        elif schema_type == "array":
            schema = enforce_no_additional_properties(schema.get("items"))

        # Handle anyOf / oneOf / allOf (rare but safe)
        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema:
                for subschema in schema[key]:
                    schema = enforce_no_additional_properties(subschema)

        if "$defs" in schema:
            for def_schema in schema["$defs"].values():
                schema = enforce_no_additional_properties(def_schema)

        return schema

def retrieve(question: str, collection: Collection, top_image=N_IMAGE) -> List[Dict]:

    query_result = collection.query(
            query_texts=[question], # Chroma embeds this text
            n_results=top_image,         # Returns top results
            where={"field": "refined_caption"}
        )
    
    # if query_result and "metadatas" in query_result and len(query_result["metadatas"])>0:
    #     image_hits = list(zip(query_result["metadatas"][0], query_result["documents"][0]))
        # for m in query_result["metadatas"][0]:
        #     image_hits.append(m)

    return query_result

def rerank_candidates(query: str, query_result: List[Dict]) -> List[Dict]:

    # ── 1. Load the reranker once (fp16 halves VRAM on GPU) ──────────────────────
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    # INPUTS
    candidates = query_result["documents"][0]   # list of 20 caption strings
    metadatas  = query_result["metadatas"][0]   # keep metadata aligned for later

    # ── 3. Rerank ────────────────────────────────────────────────────────────────
    pairs  = [[query, doc] for doc in candidates]
    scores = reranker.compute_score(pairs, batch_size=16, normalize=True)

    # Sort candidates + metadata together by score descending
    ranked = sorted(zip(scores, candidates, metadatas), reverse=True)

    # Take top-k for the LLM (3–5 is typically enough)
    TOP_K = 5
    top_docs  = [doc  for _, doc, _    in ranked[:TOP_K]]
    top_meta  = [meta for _, _,   meta in ranked[:TOP_K]]
    top_scores = [round(score, 4) for score, _, _ in ranked[:TOP_K]]

    # ── 4. Pass to your LLM ──────────────────────────────────────────────────────
    context = "\n\n".join(
        f"[{i+1}] (score: {top_scores[i]}) {doc}"
        for i, doc in enumerate(top_docs)
    )

    return ranked[:TOP_K]  # return top-k candidates with metadata for LLM input


# def download_jina_reranker_model():
#     """
#     Ensure the Jina Reranker model is downloaded locally.
#     """

#     # Download the model locally
#     model_name = "jinaai/jina-reranker-m0"
#     local_dir = os.path.join(os.getcwd(), "jina_models", "reranker")
#     os.makedirs(local_dir, exist_ok=True)

#     print(f"Downloading {model_name} to {local_dir}...")
#     HubIO().fetch(uses=model_name, target=local_dir)
#     return local_dir

# def rerank_candidates_with_jina(query: str, query_result: List[Dict]) -> List[Dict]:
#     """
#     Rerank candidates using the Jina Reranker model locally.

#     Args:
#         query (str): The user's query.
#         query_result (List[Dict]): The initial retrieval results.

#     Returns:
#         List[Dict]: The reranked candidates.
#     """
#     # Ensure the model is downloaded locally
#     local_model_path = download_jina_reranker_model()

#     # Create a Jina Flow with the local model
#     flow = Flow().add(uses=local_model_path)

#     # Extract candidates and metadata
#     candidates = query_result["documents"][0]  # list of captions
#     metadatas = query_result["metadatas"][0]  # metadata aligned with captions

#     # Prepare documents for reranking
#     docs = DocumentArray()
#     for candidate, meta in zip(candidates, metadatas):
#         image_path = meta["image_path"]
#         try:
#             with open(image_path, "rb") as img_file:
#                 image_data = img_file.read()
#             docs.append(Document(content=candidate, blob=image_data))
#         except FileNotFoundError:
#             print(f"[WARN] Image not found at path: {image_path}")
#             docs.append(Document(content=candidate))

#     # Perform reranking
#     with flow:
#         reranked_docs = flow.post(on="/rerank", inputs=docs, parameters={"query": query})

#     # Combine reranked results with metadata
#     ranked = [
#         {"document": doc.content, "metadata": meta}
#         for doc, meta in zip(reranked_docs, metadatas)
#     ]

#     return ranked

# -----------------------
# Compose LLM prompt and query LLM (example using OpenAI Chat if available)
# -----------------------
def llm_answer(question: str, candidates: List[Dict], openai_client=None) -> str:

    schema = Answer.model_json_schema()
    schema = enforce_no_additional_properties(schema)

    prompt = f"""You are an archivist for a comic panel database. 
    Answer the user's question using ONLY the following candidate panels (cite PanelID if you reference it).

    QUESTION:
    {question}

    Give a concise answer and list which panel(s) you relied on. If you do not know the answer, say "I don't know." and describe what happens in the first image.
    """

    context_blocks = []
    context_blocks.append({"type": "text", "text": prompt})
    for _, _, c in candidates:  # Loop through metadata of retrieved candidates
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
            response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Answer",
                        "schema": schema,
                        "strict": True
                        }
                },
            temperature=0.0
        )

        try:
            # Validate and parse JSON → Pydantic object
            parsed = Answer.model_validate_json(resp.choices[0].message.content)
            return parsed

        except ValidationError as e:
            print("[WARN] JSON validation failed:", e)
            print("Raw output was:", resp.choices[0].message.content)
    else:
        # If OpenAI LLM not available, return prompt + context (you can paste to any LLM).
        return prompt
    
if __name__ == "__main__":
                        
    load_dotenv()  # load environment variables from .env file
    #get collection
    ensure_dir(os.environ["CHROMA_DB_PATH"])
    client = PersistentClient(path=os.environ["CHROMA_DB_PATH"])

    # create/get image collection
    image_db = client.get_or_create_collection(name="refined_panel_captions", embedding_function=E5SmallTextEmbedder())

    user_query = input("Enter your question about the comic panels: ")

    candidates = retrieve(user_query, collection=image_db, top_image=N_IMAGE)

    reranked = rerank_candidates(user_query, candidates)

    openai_client = OpenAI()

    answer = llm_answer(user_query, reranked, openai_client=openai_client)
    print("\n=== ANSWER ===")
    print(answer)