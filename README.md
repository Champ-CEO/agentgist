# AgentGist

Agentic workflow for analysing Reddit posts

![AgentGist interface](.github/banner.png)

Features:

- Analyze and report on the latest Reddit posts in a subreddit
- 100% local and offline agentic workflow with LangGraph (functional API)
- Tool calling - agent using semantic search (embeddings) to filter posts
- Human in the loop - specify which posts to analyze
- Uses Ollama for LLM inference

## Agents

- [`Post Fetcher`](agentgist/agents/post_fetcher.py): fetches Reddit posts and associated comments
- [`Filter Query`](agentgist/agents/filter_query.py): asks the user on which posts to focus
- [`Post Analyzer`](agentgist/agents/post_analyzer.py): analyzes a given post (summary, takeaways, sentiment, etc.)
- [`Report Writer`](agentgist/agents/report_writer.py): writes a report based on the analysis of the posts and user query

## Install

Make sure you have [`uv` installed](https://docs.astral.sh/uv/getting-started/installation/).

Clone the repository:

```bash
git clone git@github.com:mlexpertio/agentgist.git .
cd agentgist
```

Install Python:

```bash
uv python install 3.12.8
```

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

Install package in editable mode:

```bash
uv pip install -e .
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

### Run Ollama

AgentGist uses Ollama for LLM inference. Watch this video to see how to install Ollama: https://www.youtube.com/watch?v=lmFCVCqOlz8

The model for tool calling and post analysis we'll use is Qwen 2.5 7B:

```bash
ollama pull qwen2.5
```

You need DeepSeek-R1 14B for writing the report:

```bash
ollama pull deepseek-r1:14b
```

Feel free to experiment with other models.

### (Optional) Groq API

You can also use models from Groq (get your API key from https://console.groq.com/keys).

Rename the `.env.example` file to `.env` and add your API key inside:

```bash
mv .env.example .env
```

Look into the [`config.py`](agentgist/config.py) file to set your preferred model.


## Run the Streamlit app

Run the app:

```bash
streamlit run app.py
```