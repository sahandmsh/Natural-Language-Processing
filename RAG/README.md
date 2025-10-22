# Retrieval-Augmented Generation (RAG) System

A modular, lightweight RAG pipeline implemented from scratch in Python.
This project demonstrates how to combine dense retrieval, cross-encoder reranking, and generative models to build a system that can retrieve, rank, and answer queries using external knowledge base.

## Key Components
1. **RAGCorpusManager** manages the creation, deduplication, and indexing of the text corpus. It handles all the preprocessing, chunking, embedding, and FAISS indexing:
* Normalizes and deduplicates input text entries.

* Splits documents into overlapping chunks using a sliding window.

* Encodes chunks using SentenceTransformers (bi-encoder).

* Creates a FAISS vector index for similarity search.

* Supports dynamic corpus updates without full reindexing.

2. **RAGAnswerRetrieval** retrieves and generates context-aware answers using a hybrid retrieval-generation approach:

* Performs bi-encoder retrieval to get top candidate passages from FAISS (e.g., top 100)

* Applies cross-encoder reranking for fine-grained similarity scoring (e.g., to find top 5 text chunks)

* Merges overlapping text chunks intelligently.

* Uses a generative model (e.g., Gemini) to extract the answer given the retrieved context.


3. **Utility Function load_hf_models** is a helper function to load Hugging Face cross-encoder tokenizer, and sentence transformer models with an authentication token.
   
4. **load_gemini_model** is a helper function to load a Gemini generative model for text generation.

5. **retrieve_squad_dataset** is a function to load Squad dataset and sample the dataset, which will be used as the knowledge base.
