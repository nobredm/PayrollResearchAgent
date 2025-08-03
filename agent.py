import os
import sys
from typing import Any

from langgraph.prebuilt import create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import MessagesState, StateGraph, END

# Depending on whether an OpenAI key is available, use a real LLM or a fake placeholder
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.chat_models.fake import FakeListChatModel
except Exception as exc:  # pragma: no cover - fail-fast in case packages missing
    raise RuntimeError("Required packages are not installed") from exc


def build_agent() -> Any:
    """Create a LangGraph ReAct agent with a Wikipedia tool."""
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return create_react_agent(llm, [wiki_tool])

    # Fallback used in environments without API access so examples still run
    llm = FakeListChatModel(responses=["This is a placeholder response."])

    def call_model(state: MessagesState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_edge("agent", END)
    graph.set_entry_point("agent")
    return graph.compile()


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What is payroll tax?"
    graph = build_agent()
    result = graph.invoke({"messages": [("user", question)]})
    final_message = result["messages"][-1].content
    print(final_message)


if __name__ == "__main__":
    main()