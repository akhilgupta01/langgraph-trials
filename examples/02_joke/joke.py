from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

simulate_exception = True


class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str


def generate_joke(state: JokeState) -> JokeState:
    topic = state["topic"]
    response = llm.invoke(
        f"Tell me a short, funny joke about {topic}. Just the joke, nothing else."
    )
    return {"joke": response.content}


def explain_joke(state: JokeState) -> JokeState:
    if simulate_exception:
        raise Exception("Something went wrong while explaining the joke.")
    joke = state["joke"]
    response = llm.invoke(
        f"Explain in short (20 words or less) why the following joke is funny:\n\n{joke}"
    )
    return {"explanation": response.content}


graph = StateGraph(JokeState)

graph.add_node("generate_joke", generate_joke)
graph.add_node("explain_joke", explain_joke)

graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "explain_joke")
graph.add_edge("explain_joke", END)

checkpointer = MemorySaver()
workflow = graph.compile(checkpointer=checkpointer, name="Joke Workflow")

if __name__ == "__main__":
    langsmith_config = {
        "tags": ["Joke", "google-genai"],
        "metadata": {"model": "gemini-2.5-flash", "temperature": 0.7},
    }

    run_config_1 = {"configurable": {"thread_id": "session_1"}}
    try:
        simulate_exception = True
        attempt_1 = workflow.invoke(
            {"topic": "pizza"}, config=run_config_1 | langsmith_config
        )
    except Exception as e:
        print("Error occurred:", e)

    # get the last checkpoint id (till where the workflow executed successfully)
    checkpoint = workflow.get_state(run_config_1)
    print("explanation:", checkpoint.values.get("explanation"))
    checkpoint_id = checkpoint.config["configurable"]["checkpoint_id"]
    print("checkpoint_id: ", checkpoint_id)

    # now we can invoke the workflow again, and it should continue from the last successful checkpoint
    simulate_exception = False
    run_config_2 = {
        "configurable": {"thread_id": "session_1", "checkpoint_id": checkpoint_id}
    }
    attempt_2 = workflow.invoke(None, config=run_config_2 | langsmith_config)
    print("explanation:", attempt_2["explanation"])

    # We can also update the state at a specific checkpoint and re-run the workflow.
    # For example, let's change the topic and see how the joke and explanation change accordingly.

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

    # Extract checkpoint before execution of node "generate_joke"
    print("\n\nExtracting checkpoint before 'generate_joke':")
    checkpoint_before_generate = state_before_node(
        workflow, run_config_1, "generate_joke"
    )
    checkpoint_to_update = checkpoint_before_generate.config["configurable"][
        "checkpoint_id"
    ]
    if checkpoint_before_generate:
        print("checkpoint id before 'generate_joke':", checkpoint_to_update)
        print("state before 'generate_joke':", checkpoint_before_generate.values)

    # Now let's update the topic in this checkpoint and re-run the workflow to see the new joke and explanation.
    update_config = {
        "configurable": {
            "thread_id": "session_1",
            "checkpoint_id": checkpoint_to_update,
            "checkpoint_ns": "",
        }
    }

    updated_state = workflow.update_state(update_config, {"topic": "pasta"})
    attempt_3 = workflow.invoke(None, config=updated_state | langsmith_config)

    # attempt_3 = workflow.invoke(None, config=config_3)
    print("attempt3:\n", attempt_3)

    print("\n\nState history:")
    state_history = workflow.get_state_history(run_config_1)
    for state in reversed(list(state_history)):
        error_messages = [
            task.error for task in state.tasks if getattr(task, "error", None)
        ]
        extracted_error = error_messages[0] if error_messages else None
        # print(state)
        print(
            "Checkpoint: ",
            state.config["configurable"]["checkpoint_id"],
            "Next node: ",
            state.next,
            "Error: ",
            extracted_error,
            "Values: ",
            state.values,
        )
