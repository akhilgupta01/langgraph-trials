# Agents Module

This module is a FastAPI application.

## Run directly

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8085
```

## Run through Gradle

From the repository root:

```bash
./gradlew :agents:run
```

If you do not have Gradle Wrapper yet, run with local Gradle:

```bash
gradle :agents:run
```
