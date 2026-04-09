# ComicRAG 📖

In this project, I implement a RAG workflow based on a vector database of comic book panels with a Gradio UI.

![ComicRAG UI](docs/gradio_gui_example.png)

## Use Case
I am implementing a RAG pipeline on an open source dataset of western, Golden Age comic books available here: https://obj.umiacs.umd.edu/comics/index.html

## Challenges
### 1. Comic books are multimodal, featuring information in both text and imagery.
### 2. I have found that off-the-shelf vision models are not effective on the types of imagery typical in western comics.
### 3. Comic narratives leave complex transitions in time, place, and action between panels for the readers' imagination.

## Current Approach
### 1. Panel Captioning
I am using GPT-4o to caption one panel at a time, aiming to capture the scene as well as the dialogue/text and the character descriptions.
### 2. Creating the Vector Database
I am using the E5 small text embedding model to embed these captions and store them in a vector database via Chroma.
### 3. Candidate Reranking
I am using the bge-reranker-v2-m3 model to perform query-aware candidate reranking based on the text captions.
### 4. Final Answering with LLM
I am using GPT-4o to examine the top candidates based on the user's query and provide a final answer.

## To Do
### 1. Improve accuracy.
Currently the reranker model can hurt performance of the system due to poor captioning. Further improvement will come from improving the captioning LLM and/or using a multimodal reranker model.
I plan to test the edge-optimized variants of Gemma 4 for captioning, and likely finetune the model on a hand-captioned high quality dataset.
### 2. Query LLM further and store a set of character descriptions so that the captions can reference specific character names.
Often names of characters are not mentioned in a given panel, so it would be useful to incorporate a multimodal Named Entity Recognition (NER) functionality to this project.
