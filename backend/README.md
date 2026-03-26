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
