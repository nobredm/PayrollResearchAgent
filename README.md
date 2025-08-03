# PayrollResearchAgent

This repository demonstrates a minimal LLM agent built with [LangGraph](https://github.com/langchain-ai/langgraph).
The agent uses a ReAct-style loop and can query Wikipedia to gather information about payroll concepts.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Provide an OpenAI API key in the `OPENAI_API_KEY` environment variable to use a real model.
   Without a key, the script falls back to a fake model that returns placeholder text.

## Usage
Run the agent with a question about payroll:

```bash
python agent.py "What is payroll tax?"
```

The final answer will be printed to the console.
