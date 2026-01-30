import asyncio
import base64
from pathlib import Path

from raganything import RAGAnything
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

# -------------------------------
# Configuration
# -------------------------------
WORKING_DIR = r"C:\Users\BabyBunny\Documents\Repos\ComicRAG\rag_anything_db"
PANELS_DIR = r"C:\Users\BabyBunny\Documents\Data\test_for_captioning\images"

# -------------------------------
# LLM + embedding configuration
# -------------------------------
def llm_func(prompt, **kwargs):
    return openai_complete_if_cache(
        model="gpt-4o",
        prompt=prompt,
        **kwargs
    )

light_rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_func,
    embedding_func=openai_embed,  # text-embedding-3-large
)

# -------------------------------
# Vision model function
# -------------------------------
def vision_model_func(prompt, image_data=None, **kwargs):
    """
    image_data is base64-encoded image
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ],
        }
    ]

    return openai_complete_if_cache(
        model="gpt-4o",
        messages=messages,
        **kwargs
    )

# -------------------------------
# Initialize RAGAnything
# -------------------------------
rag = RAGAnything(
    lightrag=light_rag,
    vision_model_func=vision_model_func,
)

# -------------------------------
# Ingest image-only panels
# -------------------------------
async def ingest_panels():
    for img_path in Path(PANELS_DIR).glob("*.jpg"):
        with open(img_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        # IMPORTANT: we give NO text, only image
        await rag.process_document_complete(
            file_path=None,
            multimodal_content={
                "images": [
                    {
                        "mime": "image/jpeg",
                        "data": image_b64,
                    }
                ],
                "metadata": {
                    "filename": img_path.name,
                },
            },
            output_dir=WORKING_DIR,
        )

    print("✅ Ingestion complete")

# -------------------------------
# Query
# -------------------------------
async def query_example():
    answer = await rag.aquery(
        "Who is the villain of this comic and what does he do?",
        mode="hybrid"
    )
    print("\n=== ANSWER ===\n")
    print(answer)

# -------------------------------
# Run
# -------------------------------
async def main():
    await ingest_panels()
    await query_example()

if __name__ == "__main__":
    asyncio.run(main())
