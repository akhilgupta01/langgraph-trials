"""
Reusable Data Extraction Agent.

Extracts structured information from documents using a three-phase LangGraph workflow:
1. Extract – initial data extraction from cached documents
2. Review  – validate extracted data for completeness and accuracy
3. Rework  – refine extraction based on review feedback

The review-rework cycle repeats until no major issues remain or the maximum
iteration count (default 3) is reached.
"""

import json
import logging
import time
from typing import Any, Type, TypedDict, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, CreateCachedContentConfig, Part
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MODEL_NAME = "gemini-2.5-flash"
MAX_REVIEW_CYCLES = 3

DEFAULT_PROMPTS: dict[str, str] = {
    "extract": (
        "You are an expert data extractor. Extract structured information from the "
        "provided documents accurately and thoroughly. Capture every relevant detail."
    ),
    "review": (
        "You are a meticulous data reviewer. Critically review the information present in the provided data."
        "Look for any ambiguities, contradictions, incomplete or missing fields, and values that seem implausible or "
        "internally inconsistent. If there are major issues that require rework, "
        "set has_major_issues to true and describe them clearly. If the data looks "
        "correct and complete, set has_major_issues to false."
    ),
    "rework": (
        "You are an expert data extractor. Refine the previously extracted data "
        "based on the review comments. Address each comment and improve the output "
        "while preserving information that was already correct."
    ),
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DocumentMetadata(BaseModel):
    """Metadata for a document to be processed."""

    name: str
    path: str
    mime_type: str


class ReviewResult(BaseModel):
    """Output of the review step."""

    has_major_issues: bool
    comments: str


class ExtractionState(TypedDict):
    """Internal graph state for the extraction workflow."""

    extracted_data: dict[str, Any] | None
    review_comments: str | None
    review_has_major_issues: bool
    iteration_count: int


# ---------------------------------------------------------------------------
# File upload & context-caching helpers
# ---------------------------------------------------------------------------


def _upload_files(documents: list[DocumentMetadata]) -> list[Any]:
    """Upload documents to Google servers and wait until processed."""
    client = genai.Client()
    uploaded: list[Any] = []
    for doc in documents:
        f = client.files.upload(file=doc.path)
        while f.state.name == "PROCESSING":
            time.sleep(2)
            f = client.files.get(name=f.name)
        uploaded.append(f)
    return uploaded


def _create_cache(
    uploaded_files: list[Any],
    system_instruction: str,
    model_name: str,
) -> Any:
    """Create a context cache containing all uploaded files."""
    client = genai.Client()

    if len(uploaded_files) == 1:
        contents = [uploaded_files[0]]
    else:
        contents = [
            Content(
                role="user",
                parts=[
                    Part.from_uri(file_uri=f.uri, mime_type=f.mime_type)
                    for f in uploaded_files
                ],
            )
        ]

    return client.caches.create(
        model=model_name,
        config=CreateCachedContentConfig(
            display_name="DataExtractor Cache",
            system_instruction=system_instruction,
            contents=contents,
            ttl="300s",
        ),
    )


def _cleanup(uploaded_files: list[Any], cache: Any) -> None:
    """Best-effort cleanup of uploaded files and cache."""
    client = genai.Client()
    try:
        client.caches.delete(name=cache.name)
    except Exception:
        logger.debug("Failed to delete cache %s", cache.name)
    for f in uploaded_files:
        try:
            client.files.delete(name=f.name)
        except Exception:
            logger.debug("Failed to delete uploaded file %s", f.name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def invoke_data_extractor(
    documents: list[DocumentMetadata],
    extraction_model: Type[T],
    system_prompts: dict[str, str] | None = None,
    model_name: str = MODEL_NAME,
    max_review_cycles: int = MAX_REVIEW_CYCLES,
) -> T:
    """Run the data extraction agent and return an instance of *extraction_model*.

    Args:
        documents: One or more documents to extract information from.
        extraction_model: Pydantic model class defining the desired output schema.
        system_prompts: Optional overrides keyed by ``extract``, ``review``,
            ``rework``.
        model_name: Gemini model identifier.
        max_review_cycles: Maximum review-rework iterations (default 3).

    Returns:
        Populated instance of *extraction_model*.
    """
    if not documents:
        raise ValueError("At least one document is required")

    prompts = {**DEFAULT_PROMPTS, **(system_prompts or {})}
    schema_json = json.dumps(extraction_model.model_json_schema(), indent=2)

    # -- Upload files & create a shared context cache ----------------------
    cache_system_instruction = (
        "You are an expert at analysing documents and extracting structured "
        "information. You have access to the provided documents for reference."
    )
    uploaded_files = _upload_files(documents)
    cache = _create_cache(uploaded_files, cache_system_instruction, model_name)

    try:
        cached_llm = ChatGoogleGenerativeAI(model=model_name, cached_content=cache.name)

        # -- Graph node functions (closures over cached_llm & prompts) -----

        def extract_node(state: ExtractionState) -> dict[str, Any]:
            structured_llm = cached_llm.with_structured_output(extraction_model)
            message = HumanMessage(
                content=(
                    f"{prompts['extract']}\n\n"
                    "Extract structured information from the provided documents.\n\n"
                    f"Output schema:\n{schema_json}"
                )
            )
            result = structured_llm.invoke([message])
            return {"extracted_data": result.model_dump(), "iteration_count": 0}

        review_llm = ChatGoogleGenerativeAI(model=model_name)

        def review_node(state: ExtractionState) -> dict[str, Any]:
            structured_llm = review_llm.with_structured_output(ReviewResult)
            extracted_json = json.dumps(state["extracted_data"], indent=2, default=str)
            message = HumanMessage(
                content=(
                    f"{prompts['review']}\n\n"
                    "Critically review the following extracted data against "
                    f"the expected output schema.\n\n"
                    f"Extracted data:\n{extracted_json}\n\n"
                    f"Expected output schema:\n{schema_json}\n\n"
                    "Identify any ambiguities, contradictions, incomplete or "
                    "missing fields, and values that seem implausible or "
                    "internally inconsistent."
                )
            )
            result = structured_llm.invoke([message])
            return {
                "review_comments": result.comments,
                "review_has_major_issues": result.has_major_issues,
                "iteration_count": state["iteration_count"] + 1,
            }

        def rework_node(state: ExtractionState) -> dict[str, Any]:
            structured_llm = cached_llm.with_structured_output(extraction_model)
            extracted_json = json.dumps(state["extracted_data"], indent=2, default=str)
            message = HumanMessage(
                content=(
                    f"{prompts['rework']}\n\n"
                    f"Previously extracted data:\n{extracted_json}\n\n"
                    f"Review comments:\n{state['review_comments']}\n\n"
                    "Refine the extracted data addressing the review comments.\n\n"
                    f"Output schema:\n{schema_json}"
                )
            )
            result = structured_llm.invoke([message])
            return {"extracted_data": result.model_dump()}

        def should_continue(state: ExtractionState) -> str:
            if not state.get("review_has_major_issues"):
                return "end"
            if state["iteration_count"] >= max_review_cycles:
                return "end"
            return "rework"

        # -- Build & compile the graph -------------------------------------

        graph = StateGraph(ExtractionState)
        graph.add_node("extract", extract_node)
        graph.add_node("review", review_node)
        graph.add_node("rework", rework_node)

        graph.add_edge(START, "extract")
        graph.add_edge("extract", "review")
        graph.add_conditional_edges(
            "review",
            should_continue,
            {"rework": "rework", "end": END},
        )
        graph.add_edge("rework", "review")

        compiled = graph.compile()

        # -- Execute -------------------------------------------------------

        final_state = compiled.invoke(
            {
                "extracted_data": None,
                "review_comments": None,
                "review_has_major_issues": False,
                "iteration_count": 0,
            }
        )

        return extraction_model.model_validate(final_state["extracted_data"])

    finally:
        _cleanup(uploaded_files, cache)
