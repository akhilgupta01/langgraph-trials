---
name: ChatGoogleGenerativeAI
description: Using ChatGoogleGenerativeAI, a chat model wrapper from langchain for Google Gemini series, for various applications including file processing.
---

# Instantiation

```python

    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")
    model.invoke("Write me a ballad about LangChain")

```

# File Processing

## PDF Input

Chat with model to describe a PDF document

```python

    import base64
    from langchain.messages import HumanMessage

    pdf_bytes = open("/path/to/your/test.pdf", "rb").read()
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe the document in a sentence"},
            {
                "type": "file",
                "source_type": "base64",
                "mime_type": "application/pdf",
                "data": pdf_base64,
            },
        ]
    )
    ai_msg = model.invoke([message])

```

## File upload

You can also upload files to Google's servers and reference them by URI.
This works for PDFs, images, videos, and audio files.

```python

    import time
    from google import genai
    from langchain.messages import HumanMessage

    client = genai.Client()

    myfile = client.files.upload(file="/path/to/your/sample.pdf")
    while myfile.state.name == "PROCESSING":
    time.sleep(2)
    myfile = client.files.get(name=myfile.name)

    message = HumanMessage(
        content=[
            {"type": "text", "text": "What is in the document?"},
            {
                "type": "media",
                "file_uri": myfile.uri,
                "mime_type": "application/pdf",
            },
        ]
    )
    ai_msg = model.invoke([message])

```

## Context Caching

Context caching allows you to store and reuse content (e.g., PDFs, images) for faster processing. The cached_content parameter accepts a cache name created via the Google Generative AI API.

### Single file caching example

```python

    from google import genai
    from google.genai import types
    import time
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.messages import HumanMessage

    client = genai.Client()

    # Upload file
    file = client.files.upload(file="path/to/your/file")
    while file.state.name == "PROCESSING":
        time.sleep(2)
        file = client.files.get(name=file.name)

    # Create cache
    model = "gemini-3.1-pro-preview"
    cache = client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            display_name="Cached Content",
            system_instruction=(
                "You are an expert content analyzer, and your job is to answer "
                "the user's query based on the file you have access to."
            ),
            contents=[file],
            ttl="300s",
        ),
    )

    # Query with LangChain
    llm = ChatGoogleGenerativeAI(
        model=model,
        cached_content=cache.name,
    )
    message = HumanMessage(content="Summarize the main points of the content.")
    llm.invoke([message])

```

### Multiple file caching example

```python

    from google import genai
    from google.genai.types import CreateCachedContentConfig, Content, Part
    import time
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.messages import HumanMessage

    client = genai.Client()

    # Upload files
    file_1 = client.files.upload(file="./file1")
    while file_1.state.name == "PROCESSING":
        time.sleep(2)
        file_1 = client.files.get(name=file_1.name)

    file_2 = client.files.upload(file="./file2")
    while file_2.state.name == "PROCESSING":
        time.sleep(2)
        file_2 = client.files.get(name=file_2.name)

    # Create cache with multiple files
    contents = [
        Content(
            role="user",
            parts=[
                Part.from_uri(file_uri=file_1.uri, mime_type=file_1.mime_type),
                Part.from_uri(file_uri=file_2.uri, mime_type=file_2.mime_type),
            ],
        )
    ]
    model = "gemini-3.1-pro-preview"
    cache = client.caches.create(
        model=model,
        config=CreateCachedContentConfig(
            display_name="Cached Contents",
            system_instruction=(
                "You are an expert content analyzer, and your job is to answer "
                "the user's query based on the files you have access to."
            ),
            contents=contents,
            ttl="300s",
        ),
    )

    # Query with LangChain
    llm = ChatGoogleGenerativeAI(
        model=model,
        cached_content=cache.name,
    )
    message = HumanMessage(
        content="Provide a summary of the key information across both files."
    )
    llm.invoke([message])

```
