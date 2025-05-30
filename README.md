# IT Helpdesk Chatbot

This project is a web-based IT Helpdesk chatbot built using FastAPI, LangChain, and a local large language model (LLM) integrated via Ollama. The chatbot is capable of classifying IT-related queries, retrieving relevant answers from a knowledge base, and generating answers when no relevant FAQ is found. It also continuously improves by logging new Q&A pairs to the knowledge base.

## Features

- IT vs. Non-IT query classification using sentence embeddings.
- Semantic search over an FAQ database with FAISS vector store.
- Generative response fallback using a locally running LLM (via Ollama).
- Automatic expansion of the FAQ knowledge base.
- Maintains conversational memory using LangChain's `ConversationBufferMemory`.
- Web-based chat interface with FastAPI and Jinja2 templates.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running with a compatible model (e.g., `phi:latest`)
- A HuggingFace-compatible embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`)

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/it-helpdesk-chatbot.git
cd it-helpdesk-chatbot

