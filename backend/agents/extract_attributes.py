"""Extract all regulatory attributes in a single pass.

Usage:
    python backend/agents/extract_attributes.py

Requires a valid GOOGLE_API_KEY in .env.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from reg_model import RegulatoryAttribute, RegulatoryAttributes
from DataExtractor import DocumentMetadata, invoke_data_extractor


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


OUTPUT_ROOT = Path("output/extraction")
OUTPUT_BASENAME = "all_attributes"
MAX_EXTRACTION_LOOPS = 100


def _safe_name(text: str) -> str:
    """Convert *text* to a filesystem-safe slug."""
    return re.sub(r"[^\w]+", "_", text).strip("_").lower()


@dataclass(frozen=True)
class JsonStore:
    """Derived output paths for aggregate attribute extraction output."""

    path: Path
    basename: str

    @property
    def slug(self) -> str:
        return _safe_name(self.basename)

    @property
    def attributes_dir(self) -> Path:
        return self.path / self.slug

    @property
    def file(self) -> Path:
        return self.path / f"{self.slug}.json"

    def attribute_file(self, attribute_name: str) -> Path:
        return self.attributes_dir / f"{_safe_name(attribute_name)}.json"


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _save_payload(file: Path, attributes: list[RegulatoryAttribute]) -> None:
    payload = {
        "extractedAttributes": [attribute.model_dump() for attribute in attributes],
    }
    _save_json(file, payload)
    print(f"Saved aggregate attributes JSON -> {file}")


def _load_existing_attributes(output_paths: JsonStore) -> list[RegulatoryAttribute]:
    """Load already extracted attributes from per-attribute JSON files."""
    if not output_paths.attributes_dir.exists():
        return []

    attributes: list[RegulatoryAttribute] = []
    for file_path in sorted(output_paths.attributes_dir.glob("*.json")):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            attributes.append(RegulatoryAttribute.model_validate(payload))
        except Exception as exc:
            print(f"Warning: failed to load {file_path}: {exc}")
    return attributes


def system_prompts() -> dict[str, str]:
    return {
        "extract": (
            "You are an expert in regulatory reporting domain "
            "You are required to extract various data entities from one or more regulatory documents."
            "You would be provided with the schema of the data entity to be extracted along with the regulatory documents. "
            "Definition and purpose of the data entity:\n"
            "- Data entities to be extracted are the data attributes of a trade that need to be reported to the regulatory authorities. "
            "Guidelines for extraction:\n"
            "1. Extract at max 10 attributes that are *explicitly* defined in the documents. "
            "2. Resolve any references to other sections or documents to provide values for various attributes of the data entity. (except for citation related attributes) "
            "3. If the attribute is not explicitly defined in the documents, do not infer or guess the value based on other attributes. "
            "4. If the attribute is defined in multiple places in the documents, try to consolidate the information to provide a single answer for each attribute. "
        )
    }


def user_prompts(excluding: list[str] | None = None) -> dict[str, str]:
    excluding = excluding or []
    return {
        "extract": (
            f"Extract next set of data entities, excluding the following entities: {excluding}\n"
        )
    }


def prepare_documents_list() -> list[DocumentMetadata]:
    documents = [
        DocumentMetadata(
            name="asic-2024-rules-schedule-1-technical-guidance-v1-1-07feb25.pdf",
            path="/Users/akhilgupta/Downloads/asic-2024-rules-schedule-1-technical-guidance-v1-1-07feb25.pdf",
            mime_type="application/pdf",
        )
    ]

    return documents


if __name__ == "__main__":
    documents = prepare_documents_list()
    output_paths = JsonStore(path=OUTPUT_ROOT, basename=OUTPUT_BASENAME)

    existing_attribute_models = _load_existing_attributes(output_paths)
    if existing_attribute_models:
        print(
            f"Loaded {len(existing_attribute_models)} existing attribute(s) from "
            f"{output_paths.attributes_dir}/"
        )

    merged_by_name: dict[str, RegulatoryAttribute] = {
        attribute.name: attribute for attribute in existing_attribute_models
    }
    total_new_attributes = 0

    for iteration in range(1, MAX_EXTRACTION_LOOPS + 1):
        excluding_names = list(merged_by_name.keys())
        print(
            f"\nIteration {iteration}: extracting next batch "
            f"(excluding {len(excluding_names)} existing attribute(s))..."
        )

        attributes_result = invoke_data_extractor(
            documents=documents,
            extraction_model=RegulatoryAttributes,
            system_prompts=system_prompts(),
            user_prompts=user_prompts(excluding=excluding_names),
            use_content_cache=False,
        )

        new_unique_attributes: list[RegulatoryAttribute] = []
        for attribute in attributes_result.attributes:
            if attribute.name in merged_by_name:
                continue
            merged_by_name[attribute.name] = attribute
            new_unique_attributes.append(attribute)

        if not new_unique_attributes:
            print("No new attributes returned; stopping extraction loop.")
            break

        total_new_attributes += len(new_unique_attributes)
        print(
            f"New unique attributes in iteration {iteration}: {len(new_unique_attributes)}"
        )
        for index, attribute in enumerate(new_unique_attributes, start=1):
            _print_attribute(index, attribute)

        merged_attributes = list(merged_by_name.values())
        _save_payload(output_paths.file, merged_attributes)
        for attribute in merged_attributes:
            _save_json(
                output_paths.attribute_file(attribute.name), attribute.model_dump()
            )
    else:
        print(
            f"Reached loop safety limit ({MAX_EXTRACTION_LOOPS}) iterations; "
            "stopping to avoid an infinite loop."
        )

    merged_attributes = list(merged_by_name.values())
    _save_payload(output_paths.file, merged_attributes)
    for attribute in merged_attributes:
        _save_json(output_paths.attribute_file(attribute.name), attribute.model_dump())

    print(
        f"Added {total_new_attributes} new attribute(s) in this run. "
        f"Saved {len(merged_attributes)} total per-attribute JSON file(s) "
        f"-> {output_paths.attributes_dir}/"
    )
