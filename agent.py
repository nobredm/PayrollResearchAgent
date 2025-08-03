"""Simple LLM-based agent built with LangGraph.

This script demonstrates how to construct a minimal LangGraph workflow
that routes a user's question through an LLM. If an OpenAI API key is
available, the agent will call an OpenAI model; otherwise it falls back
to a deterministic fake LLM so the example can run offline.
"""
from __future__ import annotations

from typing import Dict

from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.fake import FakeListLLM

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None  # type: ignore


def get_model():
    """Return an LLM instance.

    Uses ChatOpenAI if available, otherwise falls back to FakeListLLM so the
    script can run without external credentials.
    """
    if ChatOpenAI is not None:
        try:
            return ChatOpenAI()
        except Exception:
            pass
    return FakeListLLM(responses=["Hello from a fake LLM!"], n=1)


def build_agent():
    """Compile and return a LangGraph app that answers a question."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ])
    model = get_model()
    chain = prompt | model | StrOutputParser()

    def call_model(state: Dict[str, str]) -> Dict[str, str]:
        answer = chain.invoke({"question": state["question"]})
        return {"answer": answer}

    workflow = StateGraph(dict)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    workflow.add_edge("model", END)
    return workflow.compile()


if __name__ == "__main__":
    app = build_agent()
    result = app.invoke({"question": "What is LangGraph?"})
    print(result["answer"])
