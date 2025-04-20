Intelligent IT Helpdesk Chatbot
Built with FastAPI | LangChain | Ollama | HuggingFace Transformers | FAISS

🧠 Overview
This project implements an AI-powered IT Helpdesk chatbot capable of answering user queries with high accuracy and relevance. It combines Large Language Models (LLMs), semantic search, and contextual memory to simulate human-like IT support responses.

⚙️ Tech Stack
FastAPI – For building a high-performance web interface and serving the chatbot.

LangChain – Handles LLM interaction, memory management, and conversational flow.

Ollama – Local deployment of lightweight LLMs (e.g., phi) for faster, secure inference.

HuggingFace Transformers – Used for generating semantic embeddings of queries and FAQ documents.

FAISS – Vector store for fast, similarity-based retrieval of relevant answers from a custom knowledge base.

Jinja2 – Templating engine for rendering interactive frontend chat interface.

🔍 Features
Conversational Memory: Remembers context from prior messages using LangChain’s ConversationBufferMemory.

FAQ Semantic Retrieval: Embeds and retrieves relevant Q&A pairs using FAISS and HuggingFace embeddings.

IT Query Classification: Differentiates between IT and non-IT queries to keep responses focused.

Dynamic Learning: If a question is not found in the knowledge base, the LLM generates a new answer and logs it for future use.

Web UI: Interactive chat interface built using FastAPI and Jinja2 templates.

📁 Project Structure
graphql
Copy
Edit
├── chatbot.py              # Core logic: classification, LLM response, memory, retrieval
├── main.py                 # FastAPI app with GET/POST routes
├── faq.txt                 # Text-based knowledge base (FAQ)
├── templates/
│   └── chat.html           # Jinja2 frontend template
├── README.md               # Project documentation
🚀 How to Run
Clone the Repository

bash
Copy
Edit
git clone https://github.com/your-username/it-helpdesk-chatbot.git
cd it-helpdesk-chatbot
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Start the Server

bash
Copy
Edit
uvicorn main:app --reload
Open in Browser
Visit http://127.0.0.1:8000 to interact with the chatbot.

📌 Notes
Ensure Ollama is running and your selected model (e.g., phi) is available locally.

faq.txt is continuously updated as the bot learns new Q&A pairs.

