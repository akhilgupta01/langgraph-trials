from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .ChatAgent import invoke_chat

chat_router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "user_session_1"


class ChatResponse(BaseModel):
    answer: str


@chat_router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    answer = invoke_chat(user_message=payload.message, thread_id=payload.thread_id)
    return ChatResponse(answer=answer)
