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
from typing import Any, Type, TypedDict, TypeVar

from dotenv import load_dotenv
from google.genai.types import Content, CreateCachedContentConfig, Part
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from model_provider import build_chat_model, create_genai_client, default_model_name

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MODEL_NAME = default_model_name()
MAX_REVIEW_CYCLES = 3

DEFAULT_PROMPTS: dict[str, str] = {
    "extract": (
        "You are an expert data extractor. Extract structured information from the "
        "provided documents accurately and thoroughly. Capture every relevant detail.\n\n"
        "Resolve any references to other pages and sections before extracting data.\n\n"
        "<output_schema>\n{schema_json}\n</output_schema>"
    ),
    "review": (
        "You are a meticulous data reviewer. Critically review the information present "
        "in the provided data. Critically review the following extracted data against "
        "the expected output schema. Look for any ambiguities, contradictions, incomplete "
        "or missing fields, and values that seem implausible or internally inconsistent. "
        "Ensure that the values of all fields are fully resolved and not just references to "
        "other sections or documents (except for citations where it is actually needed)."
        "If there are major issues that require rework, set has_major_issues to true "
        "and describe them clearly. If the data looks correct and complete, set "
        "has_major_issues to false.\n\n"
        "<extracted_data>\n{extracted_json}\n\n</extracted_data>"
        "<expected_output_schema>\n{schema_json}\n\n</expected_output_schema>"
    ),
    "rework": (
        "You are an expert data extractor. Refine the previously extracted data "
        "based on the review comments. Address each comment and improve the output "
        "while preserving information that was already correct.\n\n"
        "<previously_extracted_data>\n{extracted_json}\n\n</previously_extracted_data>"
        "<review_comments>\n{review_comments}\n\n</review_comments>"
        "Refine the extracted data addressing the review comments.\n\n"
        "<output_schema>\n{schema_json}\n\n</output_schema>"
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
# Document loading & context-caching helpers
# ---------------------------------------------------------------------------


def _parts_from_documents(documents: list[DocumentMetadata]) -> list[Part]:
    """Load local files and convert them into Gemini parts."""
    document_parts: list[Part] = []
    for doc in documents:
        with open(doc.path, "rb") as f:
            data = f.read()
        document_parts.append(Part.from_bytes(data=data, mime_type=doc.mime_type))
    return document_parts


def _create_cache(
    document_parts: list[Part],
    system_instruction: str,
    model_name: str,
) -> Any:
    """Create a context cache containing all document parts."""
    client = create_genai_client()

    if len(document_parts) == 1:
        contents = [Content(role="user", parts=[document_parts[0]])]
    else:
        contents = [
            Content(
                role="user",
                parts=document_parts,
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


def _cleanup(cache: Any | None) -> None:
    """Best-effort cleanup of cache resource."""
    if cache is None:
        return

    client = create_genai_client()
    try:
        client.caches.delete(name=cache.name)
    except Exception:
        logger.debug("Failed to delete cache %s", cache.name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def invoke_data_extractor(
    documents: list[DocumentMetadata],
    extraction_model: Type[T],
    system_prompts: dict[str, str] | None = None,
    model_name: str = MODEL_NAME,
    max_review_cycles: int = MAX_REVIEW_CYCLES,
    use_content_cache: bool = True,
) -> T:
    """Run the data extraction agent and return an instance of *extraction_model*.

    Args:
        documents: One or more documents to extract information from.
        extraction_model: Pydantic model class defining the desired output schema.
        system_prompts: Optional overrides keyed by ``extract``, ``review``,
            ``rework``.
        model_name: Model identifier for Gemini API or Vertex AI.
        max_review_cycles: Maximum review-rework iterations (default 3).
        use_content_cache: Enable Gemini cached content for document context.
            If false, document contents are sent with every extract/rework call.

    Returns:
        Populated instance of *extraction_model*.
    """
    if not documents:
        raise ValueError("At least one document is required")

    prompts: dict[str, str] = {**DEFAULT_PROMPTS, **(system_prompts or {})}
    schema_json = json.dumps(extraction_model.model_json_schema(), indent=2)
    document_parts = _parts_from_documents(documents)

    # -- Optionally create a shared context cache --------------------------
    system_instruction = (
        "You are an expert at analysing documents and extracting structured "
        "information. You have access to the provided documents for reference."
    )
    cache = (
        _create_cache(document_parts, system_instruction, model_name)
        if use_content_cache
        else None
    )

    try:
        extract_llm = build_chat_model(
            model_name=model_name,
            cached_content=cache.name if cache is not None else None,
        )
        review_llm = build_chat_model(model_name=model_name)

        def _build_prompt(template: str, **kwargs: Any) -> HumanMessage:
            prompt = template.format(**kwargs)
            if cache is not None:
                return HumanMessage(content=prompt)
            return HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    *document_parts,
                ]
            )

        # -- Graph node functions (closures over extraction/rework llm) -----

        def extract_node(state: dict[str, Any]) -> dict[str, Any]:
            structured_llm = extract_llm.with_structured_output(extraction_model)
            message = _build_prompt(prompts["extract"], schema_json=schema_json)
            result = structured_llm.invoke([message])
            return {"extracted_data": result.model_dump(), "iteration_count": 0}

        def review_node(state: ExtractionState) -> dict[str, Any]:
            structured_llm = review_llm.with_structured_output(ReviewResult)
            extracted_json = json.dumps(state["extracted_data"], indent=2, default=str)
            kwargs = {
                "extracted_json": extracted_json,
                "schema_json": schema_json,
            }
            message = _build_prompt(prompts["review"], **kwargs)
            result = structured_llm.invoke([message])
            return {
                "review_comments": result.comments,
                "review_has_major_issues": result.has_major_issues,
                "iteration_count": state["iteration_count"] + 1,
            }

        def rework_node(state: ExtractionState) -> dict[str, Any]:
            structured_llm = extract_llm.with_structured_output(extraction_model)
            extracted_json = json.dumps(state["extracted_data"], indent=2, default=str)
            kwargs = {
                "extracted_json": extracted_json,
                "review_comments": state["review_comments"],
                "schema_json": schema_json,
            }
            message = _build_prompt(prompts["rework"], **kwargs)
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
        _cleanup(cache)
