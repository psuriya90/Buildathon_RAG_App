# 🤖 RAG Document Assistant

## Buildathon Project

A Retrieval-Augmented Generation (RAG) based document assistant that allows users to upload **PDF, TXT, and DOCX documents** and ask questions using a conversational interface.

The application retrieves relevant information from uploaded documents using **Hybrid Search (FAISS + BM25)**, improves the retrieved results using a **Cross-Encoder Reranker**, and generates accurate answers using an **LLM**.

If the user's question is not relevant to the uploaded document, the application automatically performs a **Web Search** instead of generating an unsupported or hallucinated answer.

---

# 📌 Problem Statement

Traditional AI chatbots may generate incorrect or hallucinated answers when they do not have sufficient information about a specific document.

Users often need to:

- Read large documents manually
- Search for specific information
- Compare information across document sections
- Ask questions in natural language
- Avoid AI-generated answers that are not supported by the document
- Obtain up-to-date information when the document does not contain the answer

The goal of this project is to build an intelligent document assistant that can retrieve relevant information from user-provided documents and provide reliable answers.

---

# 💡 Proposed Solution

The **RAG Document Assistant** combines document retrieval and Large Language Models.

The application follows this workflow:

```text
User Uploads Document
        ↓
Document Loader
        ↓
Text Extraction
        ↓
Recursive Character Text Splitter
        ↓
Document Chunks
        ↓
        ├───────────────┐
        ↓               ↓
     FAISS            BM25
   Semantic Search   Keyword Search
        ↓               ↓
        └───────┬───────┘
                ↓
        Fusion / RRF
                ↓
        Cross-Encoder
           Reranker
                ↓
       Relevance Router
          ↙          ↘
     Relevant       Irrelevant
        ↓               ↓
   RAG Answer       Web Search
        ↓               ↓
        └───────┬───────┘
                ↓
          Final Answer
                ↓
        Excel QA Logging