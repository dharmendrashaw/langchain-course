# LangChain course

This example summarizes information using Google Gemini through LangChain.

## Setup

1. Create the required API keys for Google, LangSmith, and Tavily.
2. Create a `.env` file in the project root and add your keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=your_project_name
TAVILY_API_KEY=your_tavily_api_key_here
```

3. Install dependencies and run the example:

```sh
uv sync
uv run main.py
```

> Keep your real API keys in a local `.env` file and do not commit them to version control.

### LangSmith tracking

LangSmith tracking is enabled with the `LANGSMITH_*` environment variables.

- `LANGSMITH_TRACING=true`: turns tracing on.
- `LANGSMITH_ENDPOINT`: LangSmith API endpoint.
- `LANGSMITH_API_KEY`: your LangSmith API key.
- `LANGSMITH_PROJECT`: the project name used for your traces.
