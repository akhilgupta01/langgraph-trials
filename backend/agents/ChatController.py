from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from typing import Any

from .ChatAgent import invoke_chat, invoke_chat_stream, invoke_chat_with_extraction

chat_router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "user_session_1"


class ChatResponse(BaseModel):
    answer: str


class ChatExtractionRequest(BaseModel):
    message: str
    regulatory_documents: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of documents with docName, mimeType, and content",
    )
    thread_id: str = "user_session_1"


class ChatExtractionResponse(BaseModel):
    response: str
    subgraph_result: dict[str, Any] | None = None


@chat_router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    answer = invoke_chat(user_message=payload.message, thread_id=payload.thread_id)
    return ChatResponse(answer=answer)


@chat_router.post("/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    return StreamingResponse(
        invoke_chat_stream(user_message=payload.message, thread_id=payload.thread_id),
        media_type="text/plain",
    )


@chat_router.post("/chat/extract", response_model=ChatExtractionResponse)
def chat_with_extraction(payload: ChatExtractionRequest) -> ChatExtractionResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    if not payload.regulatory_documents:
        raise HTTPException(
            status_code=400,
            detail="regulatory_documents must not be empty for extraction",
        )

    result = invoke_chat_with_extraction(
        user_message=payload.message,
        regulatory_documents=payload.regulatory_documents,
        thread_id=payload.thread_id,
    )
    return ChatExtractionResponse(
        response=result["response"], subgraph_result=result["subgraph_result"]
    )


@chat_router.post("/chat/extract/upload", response_model=ChatExtractionResponse)
async def chat_with_uploaded_documents(
    message: str = Form(...),
    thread_id: str = Form("user_session_1"),
    files: list[UploadFile] = File(...),
) -> ChatExtractionResponse:
    if not message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    if not files:
        raise HTTPException(status_code=400, detail="at least one file is required")

    temp_dir = Path(mkdtemp(prefix="chat_upload_"))
    document_refs: list[dict[str, Any]] = []

    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="file name is required")

        payload = await upload.read()
        if not payload:
            raise HTTPException(
                status_code=400,
                detail=f"uploaded file {upload.filename} is empty",
            )

        safe_name = Path(upload.filename).name
        file_path = temp_dir / f"{uuid4().hex}_{safe_name}"
        file_path.write_bytes(payload)

        document_refs.append(
            {
                "docName": safe_name,
                "mimeType": upload.content_type or "application/octet-stream",
                "path": str(file_path),
                "content": None,
            }
        )

    result = invoke_chat_with_extraction(
        user_message=message,
        regulatory_documents=document_refs,
        thread_id=thread_id,
    )

    return ChatExtractionResponse(
        response=result["response"],
        subgraph_result=result["subgraph_result"],
    )
