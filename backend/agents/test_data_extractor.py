"""
Integration test for the DataExtractor agent.

Usage:
    python backend/agents/test_data_extractor.py

Requires a valid GOOGLE_API_KEY in .env.
"""

import json
import re
import tempfile
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field

from DataExtractor import DocumentMetadata, invoke_data_extractor


class Citation(BaseModel):
    document_name: str = Field(
        description="Document name in which the attribute is defined"
    )
    page: str = Field(description="Page number or page reference from the document")
    section: str = Field(description="Section or clause reference from the document")


class ReportType(BaseModel):
    name: str = Field(description="Name of the report to be submitted to the regulator")
    description: str = Field(description="Brief description of the report")
    reportFormat: str = Field(description="Format of the report such as XML, CSV, etc.")
    reportFrequency: str = Field(
        description="Declared or inferred frequency such as daily, monthly, quarterly"
    )
    reportingAttributes: list[str] = Field(
        description="List of key reporting attributes that are required to be included in the report"
    )
    jurisdiction: str = Field(
        description="Regulatory jurisdiction such as ASIC or EMIR"
    )
    assetClasses: list[str] = Field(
        description="Applicable asset classes, such as Equity, FX, Interest Rate, etc."
    )
    citations: list[Citation] = Field(
        description="Supporting citations with document name, page, and section"
    )


class RegulatoryAttribute(BaseModel):
    name: str = Field(
        description="Attribute name exactly as used in the regulatory text"
    )
    description: str = Field(description="Business meaning of the regulatory field")
    dataType: str = Field(description="Logical data type such as string, date, decimal")
    format: str = Field(description="Declared or inferred format such as ddMMyyyy")
    optionalityRules: list[str] = Field(
        description="Rules describing whether the attribute is required, optional or conditionally required or optional in certain contexts."
    )
    valueRules: list[str] = Field(
        description="Rules describing valid values for the attribute in certain contexts, such as enumerated values or value ranges"
    )
    citations: list[Citation] = Field(
        description="Supporting citations with document name, page, and section"
    )


class RegulatoryAttributes(BaseModel):
    attributes: list[RegulatoryAttribute] = Field(
        description="Regulatory reporting attributes extracted from the documents"
    )


class ReportTypes(BaseModel):
    attributes: list[ReportType] = Field(
        description="Regulatory report types extracted from the documents"
    )


def _print_report_type(index: int, report_type: ReportType) -> None:
    print(f"\nReport Type {index} - {report_type.name}")
    print(f"  Description:   {report_type.description}")
    print(f"  Format:        {report_type.reportFormat}")
    print(f"  Frequency:     {report_type.reportFrequency}")
    print(f"  Jurisdiction:  {report_type.jurisdiction}")
    print(f"  Asset Classes: {', '.join(report_type.assetClasses)}")
    print("  Citations:")
    for citation in report_type.citations:
        print(
            f"    - document={citation.document_name}, "
            f"page={citation.page}, "
            f"section={citation.section}"
        )


def _print_attribute(index: int, attribute: RegulatoryAttribute) -> None:
    print(f"\n  Attribute {index} - {attribute.name}")
    print(f"    Description:   {attribute.description}")
    print(f"    Data Type:     {attribute.dataType}")
    print(f"    Format:        {attribute.format}")
    print(f"    Optionality Rules:  {', '.join(attribute.optionalityRules)}")
    print(f"    Value Rules:  {', '.join(attribute.valueRules)}")
    print("    Citations:")
    for citation in attribute.citations:
        print(
            f"      - document={citation.document_name}, "
            f"page={citation.page}, "
            f"section={citation.section}"
        )


ATTRIBUTE_BATCH_SIZE = 5
OUTPUT_ROOT = Path("output/extraction")


def _safe_name(text: str) -> str:
    """Convert *text* to a filesystem-safe slug."""
    return re.sub(r"[^\w]+", "_", text).strip("_").lower()


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _attribute_json_exists(report_dir: Path, attribute_name: str) -> bool:
    """Check if attribute JSON file already exists."""
    attr_file = report_dir / f"{_safe_name(attribute_name)}.json"
    return attr_file.exists()


def _load_attribute_from_json(
    report_dir: Path, attribute_name: str
) -> RegulatoryAttribute | None:
    """Load a RegulatoryAttribute from its JSON file if it exists."""
    attr_file = report_dir / f"{_safe_name(attribute_name)}.json"
    if not attr_file.exists():
        return None
    try:
        data = json.loads(attr_file.read_text(encoding="utf-8"))
        return RegulatoryAttribute.model_validate(data)
    except Exception as e:
        print(f"    Warning: failed to load {attr_file}: {e}")
        return None


def _save_report_payload(
    report_file: Path,
    report_type: ReportType,
    attributes: list[RegulatoryAttribute],
) -> None:
    report_payload = {
        **report_type.model_dump(),
        "extractedAttributes": [attribute.model_dump() for attribute in attributes],
    }
    _save_json(report_file, report_payload)


def _batched(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield successive slices of *items* of length *size*."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _build_attribute_extract_prompt(
    report_type: ReportType, attribute_names: list[str]
) -> str:
    names_bullet = "\n".join(f"  - {n}" for n in attribute_names)
    return (
        f"You are extracting regulatory reporting data attributes for Report Type: "
        f"'{report_type.name}' which is described as "
        f"<description>{report_type.description}</description> "
        "from one or more regulatory documents. "
        "Extract details ONLY for the following attributes:\n"
        f"{names_bullet}\n\n"
        "Preserve the exact field name as used in the regulation. For each attribute, "
        "include one or more citations with document_name, page, and section."
    )


def main() -> None:
    documents = [
        DocumentMetadata(
            name="asic-2024-rules-schedule-1-technical-guidance-v1-1-07feb25.pdf",
            path="/Users/akhilgupta/Downloads/asic-2024-rules-schedule-1-technical-guidance-v1-1-07feb25.pdf",
            mime_type="application/pdf",
        )
    ]

    # -----------------------------------------------------------------------
    # Step 1: Extract report types
    # -----------------------------------------------------------------------
    prompts_for_report_types = {
        "extract": (
            "You are extracting regulatory report types from one or "
            "more regulatory documents. Return every clearly defined report "
            "type together with citations. Use the exact report type as "
            "used in the regulation text. For each report type, include one or more "
            "citations with document_name, page, and section."
        )
    }

    print("Step 1 — Extracting regulatory report types...")
    report_types_result = invoke_data_extractor(
        documents=documents,
        extraction_model=ReportTypes,
        system_prompts=prompts_for_report_types,
        use_content_cache=False,
    )

    print("\n--- Extracted Regulatory Report Types ---")
    for index, report_type in enumerate(report_types_result.attributes, start=1):
        _print_report_type(index, report_type)

    # -----------------------------------------------------------------------
    # Step 2: For each report type, extract its reporting attributes in
    #         batches of ATTRIBUTE_BATCH_SIZE
    # -----------------------------------------------------------------------
    print(
        "\n\nStep 2 — Extracting attributes per report type (in batches of "
        f"{ATTRIBUTE_BATCH_SIZE})..."
    )

    for report_type in report_types_result.attributes:
        print(f"\n{'='*60}")
        print(f"Report type: {report_type.name}")
        attribute_names = report_type.reportingAttributes
        print(
            f"  {len(attribute_names)} attribute(s) to extract, "
            f"batching in groups of {ATTRIBUTE_BATCH_SIZE}"
        )
        print(f"{'='*60}")

        all_attributes: list[RegulatoryAttribute] = []
        report_slug = _safe_name(report_type.name)
        report_dir = OUTPUT_ROOT / report_slug
        report_file = OUTPUT_ROOT / f"{report_slug}.json"

        # Save report metadata immediately after report-type extraction.
        _save_report_payload(report_file, report_type, all_attributes)
        print(f"  Saved report type JSON → {report_file}")

        for batch_num, batch in enumerate(
            _batched(attribute_names, ATTRIBUTE_BATCH_SIZE), start=1
        ):
            # Check which attributes in this batch already have JSON files
            to_extract = []
            already_cached = []
            for attr_name in batch:
                if _attribute_json_exists(report_dir, attr_name):
                    already_cached.append(attr_name)
                else:
                    to_extract.append(attr_name)

            if already_cached:
                print(
                    f"\n  Batch {batch_num} — {len(already_cached)} cached, "
                    f"{len(to_extract)} to extract"
                )
                for attr_name in already_cached:
                    attr = _load_attribute_from_json(report_dir, attr_name)
                    if attr:
                        all_attributes.append(attr)
                        print(f"    ✓ Loaded cached: {attr_name}")
            else:
                print(f"\n  Batch {batch_num} — extracting: {', '.join(batch)}")

            if to_extract:
                prompts_for_attributes = {
                    "extract": _build_attribute_extract_prompt(report_type, to_extract)
                }
                batch_result = invoke_data_extractor(
                    documents=documents,
                    extraction_model=RegulatoryAttributes,
                    system_prompts=prompts_for_attributes,
                    use_content_cache=False,
                )
                all_attributes.extend(batch_result.attributes)

                for attribute in batch_result.attributes:
                    attr_file = report_dir / f"{_safe_name(attribute.name)}.json"
                    _save_json(attr_file, attribute.model_dump())
                print(f"    Extracted {len(batch_result.attributes)} new attribute(s)")
            else:
                print(
                    f"    All {len(already_cached)} attributes in batch already cached"
                )

            _save_report_payload(report_file, report_type, all_attributes)
            print(
                f"  Refreshed report JSON with {len(all_attributes)} total attributes"
            )

        print(
            f"\n  Total attributes extracted for '{report_type.name}': "
            f"{len(all_attributes)}"
        )
        for index, attribute in enumerate(all_attributes, start=1):
            _print_attribute(index, attribute)

        print(f"  Saved {len(all_attributes)} attribute file(s) → {report_dir}/")


if __name__ == "__main__":
    main()
