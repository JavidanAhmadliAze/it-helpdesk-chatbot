from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import subprocess

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load FAQ data
loader = TextLoader("faq.txt", encoding="utf-8")
documents = loader.load()

# Split documents into chunks for better processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Use HuggingFace embeddings to represent text
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Set up retriever for FAQ search
retriever = vectorstore.as_retriever()


# Function to log new questions and answers into the faq.txt file
def log_new_qa_to_faq(question: str, answer: str):
    with open("faq.txt", "a", encoding="utf-8") as faq_file:
        faq_file.write(f"Q: {question}\nA: {answer}\n\n")


# Define the response function to call Ollama's CLI
def get_bot_response(user_input: str) -> str:
    # Check if the user input exists in the faq.txt file
    with open("faq.txt", "r", encoding="utf-8") as faq_file:
        faq_content = faq_file.read()
        if user_input.lower() in faq_content.lower():
            # If the question exists in the FAQ, retrieve the corresponding answer
            # You can improve this part with better matching if needed
            start_idx = faq_content.lower().find(user_input.lower())
            answer_start = faq_content.find("A:", start_idx)
            answer_end = faq_content.find("\n\n", answer_start)
            answer = faq_content[answer_start + 2: answer_end].strip()
            return answer

    try:
        # If the question is not in the FAQ, call Ollama's CLI with the Phi model
        result = subprocess.run(
            ["ollama", "run", "phi:latest"],
            input=user_input,
            capture_output=True,
            text=True,
            check=True
        )
        bot_reply = result.stdout.strip()

        # After getting a reply from the bot, log it in the FAQ
        log_new_qa_to_faq(user_input, bot_reply)
        return bot_reply
    except subprocess.CalledProcessError as e:
        print(f"Error during Ollama CLI execution: {e}")
        return "Sorry, I couldn't process your request at the moment."


# Store chat history
chat_history = []


# Define FastAPI routes
@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "chat_history": chat_history})


@app.post("/", response_class=HTMLResponse)
async def post_chat(request: Request, user_input: str = Form(...)):
    bot_reply = get_bot_response(user_input)

    # Append the conversation to the chat history
    chat_history.append({"user": user_input, "bot": bot_reply})

    return templates.TemplateResponse("chat.html", {"request": request, "chat_history": chat_history})
