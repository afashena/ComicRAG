# ComicRAG 📖

In this project, I am both learning the modern approaches to building vector databases and using LLMs for Retrieval Augmented Generation, and also testing this technology on the challenging use case of comic books. 

## Use Case
I am implementing a RAG pipeline on an open source dataset of western, Golden Age comic books available here: https://obj.umiacs.umd.edu/comics/index.html

## Challenges
1. Comic books are multimodal, featuring information in both text and imagery.
2. I have found that off-the-shelf vision models are not effective on the types of imagery typical in wester comics.
3. Comic narratives leave complex transitions in time, place, and action between panels for the readers' imagination.

## Current Approach
1. I am using GPT-4o to caption one panel at a time, aiming to capture the scene as well as the dialogue/text and the character descriptions.
2. I will use a text embedding model to embed these captions and store them in a vector database via Chroma.
3. I am using GPT-4o to examine the top candidates based on the user's query and provide a final answer.

## To Do
1. Improve caption accuracy.
2. Query LLM further and store a set of character descriptions so that the captions can reference specific character names (important for implementing Graph RAG later).
3. Implement Graph RAG, likely via LightRAG.
4. If possible, transition to using local LLMs via Ollama rather than OpenAI's API.
