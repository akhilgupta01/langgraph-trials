import base64
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Literal, TypedDict
from collections.abc import Iterator
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)


class ReferenceDocument(BaseModel):
    docName: str
    mimeType: str
    path: str | None = None  # Optional
    content: str | None = None  # Optional


def _prepare_documents_for_state(
    regulatory_documents: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not regulatory_documents:
        return []

    prepared: list[dict[str, Any]] = []
    for raw_doc in regulatory_documents:
        doc = ReferenceDocument.model_validate(raw_doc)
        if doc.path:
            prepared.append(
                {
                    "docName": doc.docName,
                    "mimeType": doc.mimeType,
                    "path": doc.path,
                    "content": None,
                }
            )
            continue

        if doc.content:
            suffix = Path(doc.docName).suffix
            with NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as handle:
                handle.write(doc.content)
                temp_path = handle.name

            prepared.append(
                {
                    "docName": doc.docName,
                    "mimeType": doc.mimeType,
                    "path": temp_path,
                    "content": None,
                }
            )

    return prepared


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    documents: list[dict[str, Any]]
    subgraph_result: dict[str, Any] | None
    current_mode: Literal["chat", "extract"] | None


def _route_on_user_intent(state: ChatState) -> str:
    """Determine whether user wants to chat or extract regulatory attributes."""
    if state.get("documents") and len(state["documents"]) > 0:
        return "extract_subgraph"
    return "chat_node"


def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def extract_subgraph_node(state: ChatState) -> ChatState:
    """Invoke the regulatory attribute extractor subgraph."""
    from .RegAttributesExtractor import (
        ExtractionDocument,
        invoke_regulatory_attribute_extractor,
    )

    extraction_documents: list[ExtractionDocument] = []
    for raw_doc in state["documents"]:
        ref_doc = ReferenceDocument.model_validate(raw_doc)
        if not ref_doc.path:
            continue

        path = Path(ref_doc.path)
        if not path.exists():
            continue

        payload = path.read_bytes()
        if not payload:
            continue

        extraction_documents.append(
            ExtractionDocument(
                docName=ref_doc.docName,
                mimeType=ref_doc.mimeType,
                binaryContentB64=base64.b64encode(payload).decode("ascii"),
            )
        )

    result = invoke_regulatory_attribute_extractor(documents=extraction_documents)
    return {
        "subgraph_result": result.model_dump(),
        "documents": [],
        "messages": [
            HumanMessage(
                content=f"Regulatory attributes extracted: {len(result.attributes)} attributes found."
            )
        ],
    }


checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("extract_subgraph", extract_subgraph_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    _route_on_user_intent,
    {
        "chat_node": "chat_node",
        "extract_subgraph": "extract_subgraph",
    },
)
graph.add_edge("extract_subgraph", END)

chatbot = graph.compile(checkpointer=checkpointer)


def invoke_chat(user_message: str, thread_id: str = "user_session_1") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    response = chatbot.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "documents": [],
            "subgraph_result": None,
            "current_mode": None,
        },
        config=config,
    )
    return str(response["messages"][-1].content)


def _chunk_to_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def invoke_chat_stream(
    user_message: str, thread_id: str = "user_session_1"
) -> Iterator[str]:
    config = {"configurable": {"thread_id": thread_id}}
    stream = chatbot.stream(
        {
            "messages": [HumanMessage(content=user_message)],
            "documents": [],
            "subgraph_result": None,
            "current_mode": None,
        },
        config=config,
        stream_mode="messages",
    )

    for message_chunk, _metadata in stream:
        token_text = _chunk_to_text(message_chunk)
        if token_text:
            yield token_text


def invoke_chat_with_extraction(
    user_message: str,
    regulatory_documents: list[dict[str, Any]] | None = None,
    thread_id: str = "user_session_1",
) -> dict[str, Any]:
    """Invoke chat with optional regulatory document extraction.

    If regulatory_documents are provided, extracts attributes instead of normal chat.
    Returns dict with 'response' (chat response) and 'subgraph_result' (if extraction occurred).
    """
    config = {"configurable": {"thread_id": thread_id}}
    documents = _prepare_documents_for_state(regulatory_documents)
    response = chatbot.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "documents": documents,
            "subgraph_result": None,
            "current_mode": None,
        },
        config=config,
    )
    return {
        "response": str(response["messages"][-1].content),
        "subgraph_result": response.get("subgraph_result"),
    }


if __name__ == "__main__":
    thread_id = "user_session_1"
    while True:
        user_message = input("You: ")
        if user_message.lower() in ["exit", "quit"]:
            print("Exiting chatbot.")
            break
        print("Bot:", invoke_chat(user_message=user_message, thread_id=thread_id))
