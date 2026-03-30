# High Level Requirement

I require a Data Extractor agent, which is a reusable agent that extrcts structured information from a set of documents.

## Input

The agent should accept the following as input

- Metadata information about one or more documents (for e.g. name, path, mime-type etc.)
- Pydantic model of structured information to be extracted from the documents
- Optionally, a dict of system prompts for extract, review and rework steps

## Output

- Extracted information as per the Pydantic model

## High level approach

- Extract Step: The agent should start with a first round of data extraction. This will require to read the document contents and ask the LLM to extract the information in the desired format from these documents
- Review Step: The agent should then have another step to review the output of the extract phase, this step should look for any ambiguities, contradictions, incomplete or missing infromation. Review should just try to critically review the output against the original ask. Do not use the original documents during the review
- Rework Step: The agent should then send the review comments and the last extracted information back to the LLM to refine the output based on those review comments.
- The Review - Rework cycle should happen a few times until there are no more major review comments or the cycle has run for 3 times.

## Other Considerations

- As there can be mutiple iterations with LLM, documents can be cached (ChatGoogleGenerativeAI supports that) and this cache can be used during the extract and rework phases
