import json
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings

ollama_embedding = OllamaEmbeddings(model="qllama/multilingual-e5-small", base_url="http://localhost:11434", num_gpu=1)

text_splitter = SemanticChunker(embeddings=ollama_embedding, 
                                breakpoint_threshold_type="percentile", 
                                breakpoint_threshold_amount=90)

data = json.load(open(r"C:\Users\BabyBunny\Documents\Data\test_for_captioning\panel_captions_w_plot\merged_captions.json", "r"))

docs = text_splitter.create_documents(texts=[data["summary"]])

for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print(doc.page_content)
    print("\n" + "="*50 + "\n")