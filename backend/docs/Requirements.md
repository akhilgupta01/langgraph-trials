# Backend Module Requirements

## Functional Features

### Service Health

- Expose a health endpoint to verify backend service availability.

### Chat APIs

- Provide conversational chat responses for user prompts.
- Provide streaming chat responses for real-time token delivery.
- Support chat sessions with document-assisted extraction workflows.
- Accept uploaded documents for extraction-oriented chat requests.

### Regulatory Attribute Extraction

- Extract reportable regulatory attributes from inline document text.
- Extract reportable regulatory attributes from uploaded binary documents.
- Extract all reportable attributes in a single pass without requiring prior report-type classification.
- Resume extraction runs by loading already extracted attributes from persisted output files and excluding them from subsequent extraction prompts.
- Continue iterative extraction in a loop until an iteration returns no additional attributes.
- Return structured attribute output including citations and validation details.
- Support configurable extraction rework iterations within allowed limits.

### Generic Data Extraction Agent

- Instantiate a generic extraction agent with arbitrary Pydantic extraction schemas and document lists.
- Configure the extraction model name, system prompt, and review iteration budget at runtime.
- Configure separate system and user prompts for each extraction node (extract, review, rework) at runtime.
- Support both Gemini Developer API and Vertex AI model backends, selected by runtime configuration.
- Read uploaded document files from local storage and convert them to model-ready byte parts.
- Optionally cache document context for efficient multi-step extraction, or send document contents with each extraction/rework invocation.
- Extract structured information from documents in an initial extraction pass.
- Review extracted data against source documents for ambiguities, contradictions, or missing information.
- Automatically rework and refine extracted data based on review feedback (up to a configurable cycle limit).
- Persist extraction outputs using normalized, filesystem-safe report identifiers for stable report and attribute JSON file paths.
- Clean up cached content after extraction completes when caching is enabled.

### Conversation Continuity

- Preserve conversation and extraction continuity through thread identifiers.
