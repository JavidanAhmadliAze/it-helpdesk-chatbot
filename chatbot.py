
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
from langchain_community.vectorstores import FAISS
import subprocess
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load and embed FAQ data
loader = TextLoader("faq.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

memory = ConversationBufferMemory()
llm = Ollama(model="phi:latest")
conversation = ConversationChain(llm=llm, memory=memory)


IT_EXAMPLES = [
    "How can I fix slow internet?",
    "Why is my VPN not working?",
    "How to install software on Windows?",
    "My computer won't boot.",
    "Wi-Fi keeps disconnecting, what should I do?"
]

NON_IT_EXAMPLES = [
    "What’s the weather today?",
    "Who won the football game?",
    "How to cook pasta?",
    "Tell me a joke.",
    "What’s the capital of France?"
]

GENERAL_ALLOWED = ["hey", "hello", "hi", "thanks", "bye","greetings"]

def is_general_query(query: str) -> bool:
    return query.strip().lower() in GENERAL_ALLOWED


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
IT_EMBEDDINGS = embeddings.embed_documents(IT_EXAMPLES)
NON_IT_EMBEDDINGS = embeddings.embed_documents(NON_IT_EXAMPLES)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()




# Save Q&A into faq.txt
def log_new_qa_to_faq(question: str, answer: str):
    with open("faq.txt", "a", encoding="utf-8") as faq_file:
        faq_file.write(f"Q: {question}\nA: {answer}\n\n")

 # classification to calculate content of query
def is_it_related(query: str) -> bool:
    query_emb = embeddings.embed_query(query)

    sim_it = cosine_similarity([query_emb], IT_EMBEDDINGS).flatten()
    sim_non_it = cosine_similarity([query_emb], NON_IT_EMBEDDINGS).flatten()

    avg_it = np.mean(sim_it)
    avg_non_it = np.mean(sim_non_it)

    print(f"[DEBUG] IT score: {avg_it:.2f} | NON-IT score: {avg_non_it:.2f}")

    return avg_it > avg_non_it


def get_bot_response(user_input: str) -> str:
    if not is_it_related(user_input) and not is_general_query(user_input):
        return "Please ask an IT-related question."

    retriever_results = retriever.get_relevant_documents(user_input)
    if retriever_results:
        top_doc = retriever_results[0]
        doc_emb = embeddings.embed_query(user_input)
        doc_score = cosine_similarity([doc_emb], [embeddings.embed_query(top_doc.page_content)]).flatten()[0]

        print(f"[DEBUG] FAQ similarity score: {doc_score:.2f}")

        if doc_score > 0.7:
            if "A:" in top_doc.page_content:
                return top_doc.page_content.split("A:")[1].split("Q:")[0].strip()
            return top_doc.page_content

    # If not found in FAQ, use LLM + memory
    try:
        response = conversation.run(user_input)
        log_new_qa_to_faq(user_input, response)
        return response
    except Exception as e:
        print(f"Error during LLM response: {e}")
        return "Sorry, I couldn't process your request at the moment."


