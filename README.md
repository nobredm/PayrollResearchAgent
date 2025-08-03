# PayrollResearchAgent

This repository contains an example LLM-based agent built with [LangGraph](https://github.com/langchain-ai/langgraph).
The agent demonstrates a simple workflow that sends a user's question to an LLM and returns the model's answer.

## Running the agent

Install the dependencies:

```bash
pip install langgraph langchain-core langchain-openai langchain-community
```

Execute the script:

```bash
python agent.py
```

By default the script uses a deterministic fake LLM so that it can run offline.
If an `OPENAI_API_KEY` environment variable is set, the agent will instead use OpenAI's models via `langchain-openai`.
