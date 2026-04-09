import os
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient
import torch

from utils.util import ensure_dir
from vector_db.make_db import E5SmallTextEmbedder
import RAG.rag as rag

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Retrieval sizes
N_TEXT = 5
N_IMAGE = 20
N_TOTAL = 10

class RagAnswerer():
    def __init__(self):
        ensure_dir(os.environ["CHROMA_DB_PATH"])
        self.chroma_client = PersistentClient(path=os.environ["CHROMA_DB_PATH"])
        self.image_db = self.chroma_client.get_or_create_collection(name="refined_panel_captions", embedding_function=E5SmallTextEmbedder())

    def get_answer(self, question: str) -> tuple[str, str]:

        candidates = rag.retrieve(question, collection=self.image_db, top_image=N_IMAGE)

        reranked = rag.rerank_candidates(question, candidates)

        openai_client = OpenAI()

        answer = rag.llm_answer(question, reranked, openai_client=openai_client)

        for _, _, candidate_meta in reranked:
            if Path(candidate_meta["source_img"]).stem == answer.page_panel:  # Adjust extension if needed
                top_image_path = candidate_meta["source_img"]
                break

        return answer.answer_text, top_image_path



def main(rag_answerer: RagAnswerer):
    with gr.Blocks() as demo:
        gr.Markdown("## ⚡ComicRAG: A Retrieval-Augmented Generation System for Comic Panels")
        with gr.Row():
            question_input = gr.Textbox(label="Enter your question about the comic panels:")
            answer_output = gr.Textbox(label="Answer:", interactive=False)
            image_output = gr.Image(label="Top Image", interactive=False)
        submit_btn = gr.Button("Get Answer")
        submit_btn.click(fn=rag_answerer.get_answer, inputs=question_input, outputs=[answer_output, image_output])

    demo.launch(allowed_paths=[os.environ["PANEL_DIR"]])
    return

if __name__ == "__main__":
    load_dotenv()

    rag_answerer = RagAnswerer()

    main(rag_answerer)