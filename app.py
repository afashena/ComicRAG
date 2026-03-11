from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient
import torch

from utils.util import ensure_dir
from vector_db.make_db import E5SmallTextEmbedder
import RAG.rag as rag

CHROMA_DB_PATH = Path("./e5_caption_chroma_image_db")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Retrieval sizes
N_TEXT = 5
N_IMAGE = 20
N_TOTAL = 10

def get_answer(question: str) -> str:

    # Initialize ChromaDB client and collection
    ensure_dir(CHROMA_DB_PATH)
    chroma_client = PersistentClient(path=CHROMA_DB_PATH)

    # create/get image collection
    image_db = chroma_client.get_or_create_collection(name="refined_panel_captions", embedding_function=E5SmallTextEmbedder())

    candidates = rag.retrieve(question, collection=image_db, top_text=N_TEXT, top_image=N_IMAGE, max_total=N_TOTAL)

    openai_client = OpenAI()

    answer = rag.llm_answer(question, candidates, openai_client=openai_client)
    return answer



def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Comic Panel Q&A")
        with gr.Row():
            question_input = gr.Textbox(label="Enter your question about the comic panels:")
            answer_output = gr.Textbox(label="Answer:", interactive=False)
        submit_btn = gr.Button("Get Answer")
        submit_btn.click(fn=get_answer, inputs=question_input, outputs=answer_output)

    demo.launch()
    return

if __name__ == "__main__":
    load_dotenv()

    main()