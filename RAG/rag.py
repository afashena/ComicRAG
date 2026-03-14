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

from utils.util import ensure_dir, image_to_b64

from vector_db.make_db import E5SmallTextEmbedder, QwenImageEmbedder, QwenTextEmbedder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def retrieve(question: str, collection: Collection, top_text=N_TEXT, top_image=N_IMAGE, max_total=N_TOTAL) -> List[Dict]:

    query_result = collection.query(
            query_texts=[question], # Chroma embeds this text
            n_results=top_image,         # Returns top results
            where={"field": "refined_caption"}
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
    image_db = client.get_or_create_collection(name="panel_image", embedding_function=E5SmallTextEmbedder())

    user_query = input("Enter your question about the comic panels: ")

    candidates = retrieve(user_query, collection=image_db, top_text=N_TEXT, top_image=N_IMAGE, max_total=N_TOTAL)

    openai_client = OpenAI()

    answer = llm_answer(user_query, candidates, openai_client=openai_client)
    print("\n=== ANSWER ===")
    print(answer)