# Backend Module

This module is a FastAPI application.

## Run directly

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8085
```

## Run through Gradle

From the repository root:

```bash
./gradlew :backend:run
```

If you do not have Gradle Wrapper yet, run with local Gradle:

```bash
gradle :backend:run
```

## LLM Provider Configuration

The backend supports both Gemini Developer API and Vertex AI for Google models.

- `GOOGLE_LLM_PROVIDER`: `auto` (default), `gemini`, or `vertex`
- `GOOGLE_LLM_MODEL`: model name (default: `gemini-2.5-flash`)

Gemini Developer API:

- Set `GOOGLE_API_KEY`
- Optional: set `GOOGLE_LLM_PROVIDER=gemini` to force Gemini

Vertex AI:

- Set `GOOGLE_CLOUD_PROJECT` (or `VERTEX_PROJECT_ID`)
- Optional: set `GOOGLE_CLOUD_LOCATION` (or `VERTEX_LOCATION`)
- Optional: set `GOOGLE_LLM_PROVIDER=vertex` to force Vertex

When `GOOGLE_LLM_PROVIDER=auto`, Vertex is selected automatically if Vertex
configuration is detected; otherwise Gemini Developer API is used.
