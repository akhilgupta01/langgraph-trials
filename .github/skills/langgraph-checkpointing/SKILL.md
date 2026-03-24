---
name: langgraph-checkpointing
description: Adds checkpointing to a LangGraph workflow to preserve its state across multiple runs. Checkpointing can be used to add "fault tolerance" or "human in the loop" capabilities, allowing a graph to retrieve prior state and continue processing from where it left off.
---

## What a checkpoint does?

A checkpoint stores graph state between invocations so a conversation can continue over multiple turns.

In a chat bot, that means:

- Previous messages are remembered.
- Each new user message is appended to prior context.
- The model sees full conversation history for the same session.

Without checkpointing, every call behaves like a fresh conversation.

## Implementation steps

1. Create a checkpointer
   In the example below, we use `MemorySaver`, which is an in-memory checkpointer suitable for testing and demos.
   However, for production use, consider a durable checkpointer like SQLite or Postgres.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
```

2. Compile the graph with the checkpointer

```python
chatbot = graph.compile(checkpointer=checkpointer)
```

3. Provide a stable thread ID in config

```python
thread_id = "user_session_1"
config = {"configurable": {"thread_id": thread_id}}
```

4. Invoke the graph with the same config every turn

```python
response = chatbot.invoke(
    {"messages": [HumanMessage(content=user_message)]},
    config=config,
)
```

As long as `thread_id` remains the same, memory is restored for that session.

## Why thread_id matters

`thread_id` is the conversation key.

- Same `thread_id` -> continue existing chat memory.
- Different `thread_id` -> start a new independent session.

For multi-user apps, use a user-scoped and/or conversation-scoped ID, for example:

- `user_42`
- `user_42_chat_2026_03_22`

## Fault tolerance with checkpoint resume

Checkpointing is not only for chat memory. It also helps recover from partial failures in multi-step workflows.

In [02_joke/joke.py](02_joke/joke.py):

- The graph has two steps: `generate_joke` -> `explain_joke`.
- `generate_joke` succeeds and writes `joke` into state.
- `explain_joke` can fail (simulated by `simulate_exception = True`).

After the failure, the script reads the latest checkpoint and extracts `checkpoint_id`:

```python
checkpoint = workflow.get_state(config_1)
checkpoint_id = checkpoint.config["configurable"]["checkpoint_id"]
```

Then it retries with the same `thread_id` plus that `checkpoint_id`:

```python
config_2 = {
    "configurable": {"thread_id": "session_1", "checkpoint_id": checkpoint_id}
}
attempt_2 = workflow.invoke(None, config=config_2)
```

This lets the graph continue from the last successful checkpoint instead of restarting from `START`, so already completed work (the generated joke) is reused.

Practical benefits:

- Avoids re-running completed nodes after transient errors.
- Reduces duplicate LLM calls and latency.
- Makes retries deterministic for the same session state.
- Enables safer long-running or multi-step pipelines.

## Update state at a checkpoint and resume

You can also edit historical state at a selected checkpoint, then continue execution from that edited point.

In [02_joke/joke.py](02_joke/joke.py), this is done in three steps:

1. Find a checkpoint before a target node

```python
checkpoint_before_generate = state_before_node(workflow, config_1, "generate_joke")
checkpoint_to_update = checkpoint_before_generate.config["configurable"]["checkpoint_id"]
```

2. Build config for that checkpoint and update state values

```python
config_3 = {
    "configurable": {
        "thread_id": "session_1",
        "checkpoint_id": checkpoint_to_update,
        "checkpoint_ns": "",
    }
}
updated_state = workflow.update_state(config_3, {"topic": "pasta"})
```

3. Resume graph execution from the updated checkpoint

```python
attempt_3 = workflow.invoke(None, config=updated_state)
```

What this gives you:

- Branching behavior from an earlier point without rebuilding the graph.
- Ability to correct or override inputs (for example `topic`) and recompute downstream nodes.
- A practical "time-travel + replay" workflow for debugging, recovery, and human-in-the-loop edits.

Tip: Keep the same `thread_id` when replaying the same session lineage. Change `thread_id` only if you want a separate session history.

## Minimal pattern to follow

```python
checkpointer = MemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "my_session_id"}}

# Turn 1
chatbot.invoke({"messages": [HumanMessage(content="Hi")]}, config=config)

# Turn 2 (same thread_id, so memory is preserved)
chatbot.invoke({"messages": [HumanMessage(content="What did I just say?")]}, config=config)
```

## Important limitation of MemorySaver

`MemorySaver` is in-memory only.

- Memory lasts only while the Python process is running.
- Restarting the script clears checkpoints.

Use `MemorySaver` for local testing and demos.

## Production checkpointing approach

For persistence across restarts, use a durable checkpointer backend (for example SQLite or Postgres checkpointer supported by LangGraph).

Conceptually, implementation remains the same:

- Create persistent checkpointer.
- Pass it to `graph.compile(checkpointer=...)`.
- Keep using `thread_id` per session.

Only the checkpointer implementation changes; graph nodes and invoke pattern stay mostly unchanged.

## Quick verification checklist

1. Start chatbot and send: "My name is Akhil".
2. Send next message in same run: "What is my name?".
3. If checkpointing works, the model should reference the earlier message.
4. Change `thread_id` and repeat. It should behave like a new session.
5. Restart script while using `MemorySaver`. Prior memory should be gone.

## Common mistakes

- Forgetting to pass `checkpointer` in `graph.compile(...)`.
- Using a new/random `thread_id` on every request.
- Recreating graph/checkpointer per request in a web server handler.
- Expecting `MemorySaver` to survive process restarts.
