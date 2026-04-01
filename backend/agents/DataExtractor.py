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
import base64
from typing import Any, Type, TypedDict, TypeVar

from dotenv import load_dotenv
from google.genai.types import Content, CreateCachedContentConfig, Part
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate

from model_provider import build_chat_model, create_genai_client, default_model_name

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MODEL_NAME = default_model_name()
MAX_REVIEW_CYCLES = 3

NODE_KEYS = ("extract", "review", "rework")

DEFAULT_SYSTEM_PROMPTS: dict[str, str] = {
    "extract": (
        "You are an expert data extractor. Extract structured information from the "
        "provided documents accurately and thoroughly."
    ),
    "review": (
        "You are a meticulous data reviewer. Critically review extracted data for "
        "ambiguity, contradictions, missing values, and schema non-compliance."
    ),
    "rework": (
        "You are an expert data extractor. Improve previously extracted data using "
        "review feedback while preserving already-correct information."
    ),
}

DEFAULT_USER_PROMPTS: dict[str, str] = {
    "extract": (
        "Extract structured information from the provided documents. Capture every "
        "relevant detail and resolve references to other pages/sections before "
        "extracting values.\n\n"
        "<output_schema>\n{schema_json}\n</output_schema>"
    ),
    "review": (
        "Review the following extracted data against the expected output schema.\n"
        "Review Guidelines:\n"
        "- Look for ambiguities, contradictions, incomplete or missing fields\n"
        "- Flag implausible or internally inconsistent values\n"
        "- Check if values adhere to schema type/format expectations\n"
        "- Ensure values are resolved and not just references to other sections/documents "
        "(except citations where needed)\n"
        "If there are major issues that require rework, set has_major_issues=true and "
        "describe them clearly. Otherwise set has_major_issues=false.\n\n"
        "<extracted_data>\n{extracted_json}\n</extracted_data>\n"
        "<expected_output_schema>\n{schema_json}\n</expected_output_schema>"
    ),
    "rework": (
        "Refine the extracted data based on the review comments and address each "
        "comment.\n\n"
        "<previously_extracted_data>\n{extracted_json}\n</previously_extracted_data>\n"
        "<review_comments>\n{review_comments}\n</review_comments>\n"
        "<output_schema>\n{schema_json}\n</output_schema>"
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


def _base64_blocks_from_parts(document_parts: list[Part]) -> list[dict[str, str]]:
    """Convert Gemini parts into LangChain-compatible base64 file blocks."""
    file_blocks: list[dict[str, str]] = []
    for part in document_parts:
        if part.inline_data is None:
            continue
        file_blocks.append(
            {
                "type": "file",
                "source_type": "base64",
                "mime_type": part.inline_data.mime_type,
                "data": base64.b64encode(part.inline_data.data).decode("utf-8"),
            }
        )
    return file_blocks


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
    user_prompts: dict[str, str] | None = None,
    model_name: str = MODEL_NAME,
    max_review_cycles: int = MAX_REVIEW_CYCLES,
    use_content_cache: bool = False,
) -> T:
    """Run the data extraction agent and return an instance of *extraction_model*.

    Args:
        documents: One or more documents to extract information from.
        extraction_model: Pydantic model class defining the desired output schema.
        system_prompts: Optional system prompt overrides keyed by ``extract``,
            ``review``, ``rework``.
        user_prompts: Optional user prompt overrides keyed by ``extract``,
            ``review``, ``rework``.
        model_name: Model identifier for Gemini API or Vertex AI.
        max_review_cycles: Maximum review-rework iterations (default 3).
        use_content_cache: Enable Gemini cached content for document context.
            If false, document contents are sent with every extract/rework call.

    Returns:
        Populated instance of *extraction_model*.
    """
    if not documents:
        raise ValueError("At least one document is required")

    resolved_system_prompts: dict[str, str] = {
        **DEFAULT_SYSTEM_PROMPTS,
        **(system_prompts or {}),
    }
    resolved_user_prompts: dict[str, str] = {**DEFAULT_USER_PROMPTS}

    # Backward compatibility: existing callers used ``system_prompts`` as the
    # only prompt channel. Keep honoring that behavior unless ``user_prompts``
    # is explicitly provided.
    if user_prompts is None and system_prompts:
        resolved_user_prompts.update(system_prompts)
    else:
        resolved_user_prompts.update(user_prompts or {})

    for key in NODE_KEYS:
        if key not in resolved_system_prompts or key not in resolved_user_prompts:
            raise ValueError(
                f"Missing prompt configuration for node '{key}'. "
                "Expected keys: extract, review, rework."
            )

    schema_json = json.dumps(extraction_model.model_json_schema(), indent=2)
    document_parts = _parts_from_documents(documents)
    base64_file_blocks = _base64_blocks_from_parts(document_parts)

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

        def _build_messages(node_key: str, **kwargs: Any) -> list[Any]:
            system_prompt = PromptTemplate.from_template(
                resolved_system_prompts[node_key]
            ).format(**kwargs)
            user_prompt = PromptTemplate.from_template(
                resolved_user_prompts[node_key]
            ).format(**kwargs)

            system_message = SystemMessage(content=system_prompt)
            if cache is not None:
                return [system_message, HumanMessage(content=user_prompt)]

            return [
                system_message,
                HumanMessage(
                    content=[
                        {"type": "text", "text": user_prompt},
                        *base64_file_blocks,
                    ]
                ),
            ]

        # -- Graph node functions (closures over extraction/rework llm) -----

        def extract_node(state: dict[str, Any]) -> dict[str, Any]:
            structured_llm = extract_llm.with_structured_output(extraction_model)
            kwargs = {"schema_json": schema_json}
            messages = _build_messages("extract", **kwargs)
            result = structured_llm.invoke(messages)
            return {"extracted_data": result.model_dump(), "iteration_count": 0}

        def review_node(state: ExtractionState) -> dict[str, Any]:
            structured_llm = review_llm.with_structured_output(ReviewResult)
            extracted_json = json.dumps(state["extracted_data"], indent=2, default=str)
            kwargs = {
                "extracted_json": extracted_json,
                "schema_json": schema_json,
            }
            messages = _build_messages("review", **kwargs)
            result = structured_llm.invoke(messages)
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
            messages = _build_messages("rework", **kwargs)
            result = structured_llm.invoke(messages)
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
