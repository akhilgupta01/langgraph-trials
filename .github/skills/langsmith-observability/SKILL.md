---
name: langsmith-observability
description: Adds observability features to a LangGraph workflow using LangSmith, allowing monitoring and tracing of its execution. Observability can be used to gain insights into workflow performance, detect anomalies, and debug issues effectively.
---

## Initial setup

Add following environment variables to your `.env` file:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT='https://api.smith.langchain.com'
LANGCHAIN_API_KEY='your_langchain_api_key'
LANGCHAIN_PROJECT='your_langchain_project_name'
```

## Custom run name, tags, and metadata

Use LangSmith run configuration to improve trace searchability and debugging context.

### 1. Add run config per invocation

```python
langsmith_config = {
	"run_name": "Joke Factory",
	"tags": ["joke", "google-genai", "demo"],
	"metadata": {
		"model": "gemini-2.5-flash",
		"temperature": 0.7,
		"feature": "checkpoint-replay",
	},
}

run_config = {"configurable": {"thread_id": "session_1"}}
result = workflow.invoke({"topic": "pizza"}, config=run_config | langsmith_config)
```

### 2. Set defaults at compile time (optional)

If you want all runs to carry the same labels by default, pass them when compiling:

```python
workflow = graph.compile(
	checkpointer=checkpointer,
	name="Joke Workflow",
)
```

Then keep dynamic values (like user/session info) in per-run metadata:

```python
request_context = {
	"run_name": "Joke Request",
	"tags": ["prod", "user-request"],
	"metadata": {
		"user_id": "u_123",
		"request_id": "req_456",
	},
}

result = workflow.invoke({"topic": "pasta"}, config=run_config | request_context)
```

### Practical tips

- Keep `run_name` human-readable and task-oriented (for example, `Joke Retry`, `Support Triage`).
- Use low-cardinality `tags` for filtering in LangSmith UI (environment, workflow, model family).
- Use `metadata` for structured, queryable context like IDs, model params, and experiment variants.

## Add runs with `@traceable`

Use `@traceable` when you want LangSmith to create a run for a specific helper function, tool-like operation, or checkpoint utility inside your workflow code.

### 1. Import and decorate the function

```python
from langsmith import traceable


@traceable(name="get_checkpoint_before_node")
def state_before_node(workflow: StateGraph, config: dict, node_name: str):
	snapshots = reversed(list(workflow.get_state_history(config)))
	for state in snapshots:
		next_nodes = (
			state.next if isinstance(state.next, (tuple, list)) else (state.next,)
		)
		if node_name in next_nodes:
			return state
	return None
```

This creates a LangSmith run every time `state_before_node(...)` is called.

### 2. Add fixed tags and metadata on the decorator

```python
@traceable(
	name="get_checkpoint_before_node",
	run_type="tool",
	tags=["checkpoint", "debug"],
	metadata={"feature": "checkpoint-replay"},
)
def state_before_node(workflow: StateGraph, config: dict, node_name: str):
	...
```

Use this when the function should always appear in LangSmith with the same identity and labels.

### 3. Override details dynamically per call with `langsmith_extra`

```python
checkpoint_before_generate = state_before_node(
	workflow,
	run_config_1,
	"generate_joke",
	langsmith_extra={
		"name": "checkpoint_lookup_generate_joke",
		"tags": ["session-1", "manual-replay"],
		"metadata": {
			"thread_id": "session_1",
			"target_node": "generate_joke",
		},
	},
)
```

Use `langsmith_extra` when run details depend on the current request, node, thread, or experiment.

### When to use `@traceable` vs graph config

- Use graph invocation config for the main LangGraph run.
- Use `@traceable` for nested helper runs that you want to inspect separately in LangSmith.
- Use decorator arguments for stable defaults.
- Use `langsmith_extra` for per-call overrides.
