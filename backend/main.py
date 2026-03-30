from fastapi import FastAPI

from agents.ChatController import chat_router
from agents.reg_attributes_extractor import router

app = FastAPI(title="Agents API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(chat_router)
app.include_router(router)
