from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from chatbot import get_bot_response

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Store conversation history
chat_history = []

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "chat_history": chat_history})

@app.post("/", response_class=HTMLResponse)
async def post_chat(request: Request, user_input: str = Form(...)):
    bot_reply = get_bot_response(user_input)
    chat_history.append({"user": user_input, "bot": bot_reply})
    return templates.TemplateResponse("chat.html", {"request": request, "chat_history": chat_history})