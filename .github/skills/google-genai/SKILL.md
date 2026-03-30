---
name: google-genai
description: How to use native Google GenAI features like file caching.
---

## Client Instatiation

Using Gemini Developer API

```python
  from google import genai
  client = genai.Client(api_key='my-api-key')
```

Usage Vertex AI API:

```python
  from google import genai
  client = genai.Client(vertexai=True, project='my-project-id', location='us-central1')
```

## Sending Content/Files/Attachments to the Model

```python
    from google import genai
    from google.genai.types import Part

    client = genai.Client(api_key='my-api-key')

    # Part can be created from the bytes of a file
    with open("local_image.jpg", "rb") as f:
      image_bytes = f.read()

    image_part = Part.from_bytes(
        data=image_bytes,
        mime_type='image/jpeg' # Specify the correct MIME type
      )

    # Part can be created directly from a GCS uri
    video_uri = "gs://cloud-samples-data/video/animals.mp4"
    video_part = Part.from_uri(
        uri=video_uri,
        mime_type='video/mp4' # Specify the correct MIME type
      )

    # Part can also be created from a text
    text_part = Part.from_text("Summarize the video and describe the image in detail")

    # Combine with a text instruction
    contents = [image_part, video_part, text_part]
    response = client.models.generate_content(model="gemini-2.0-flash", contents=contents)
    print(response.text)

```

## Content Caching

Example below shows how to create a cached contents resource

```python

  # Create the cache.
  video_uri = "gs://cloud-samples-data/video/animals.mp4"
  contents_to_cache = [
    Content(parts = [Part.from_uri(uri=video_uri, mime_type='video/mp4')])
  ]
  content_cache = client.caches.create(
        model="gemini-2.5-flash", # Caches are model-specific
        config=CreateCachedContentConfig(
            contents=contents_to_cache,
            system_instruction=Content(parts=[Part.from_text(system_instruction)]),
            display_name="financial-reports-cache",
            ttl="3600s", # Time to live (e.g., 1 hour)
        ),
    )
  print(f"Cache created successfully. Resource name: {content_cache.name}")

  # Using the cached content while invoking the model
  response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What was the profit increase in Q1?", # Your specific, smaller prompt
    config=GenerateContentConfig(
        cached_content=content_cache.name,
    ),
  )
  print("Response 1:", response.text)

```
