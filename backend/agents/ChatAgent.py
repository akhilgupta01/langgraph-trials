from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def invoke_chat(user_message: str, thread_id: str = "user_session_1") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_message)]}, config=config
    )
    return str(response["messages"][-1].content)


if __name__ == "__main__":
    thread_id = "user_session_1"
    while True:
        user_message = input("You: ")
        if user_message.lower() in ["exit", "quit"]:
            print("Exiting chatbot.")
            break
        print("Bot:", invoke_chat(user_message=user_message, thread_id=thread_id))
