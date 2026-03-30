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
- Return structured attribute output including citations and validation details.
- Support configurable extraction rework iterations within allowed limits.

### Generic Data Extraction Agent

- Instantiate a generic extraction agent with arbitrary Pydantic extraction schemas and document lists.
- Configure the extraction model name, system prompt, and review iteration budget at runtime.
- Upload documents to Google servers and cache them for efficient multi-step extraction.
- Extract structured information from documents in an initial extraction pass.
- Review extracted data against source documents for ambiguities, contradictions, or missing information.
- Automatically rework and refine extracted data based on review feedback (up to a configurable cycle limit).
- Clean up uploaded files and cached content after extraction completes.

### Conversation Continuity

- Preserve conversation and extraction continuity through thread identifiers.
