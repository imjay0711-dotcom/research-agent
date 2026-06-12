# 🔬 Autonomous Research Agent

An AI agent that autonomously researches any topic — searching the web, reading academic papers, storing knowledge in a vector database, and generating structured research reports. Built using an agentic loop with LangGraph, demonstrating multi-tool orchestration, RAG (Retrieval Augmented Generation), and evaluation-driven development.

## 🎥 Demo

The agent runs as a Streamlit web app where you type any research topic and watch it autonomously:
1. Search the web for relevant papers
2. Download and read PDFs
3. Store and retrieve information using a vector database
4. Generate a structured markdown report with summary, findings, and references

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────┐
│   Planner    │  ← LLM decides which tool to call next
│  (LangGraph) │
└──────┬───────┘
       │
       ▼
┌─────────────┐
│ Tool Executor│  ← Runs the chosen tool
└──────┬───────┘
       │
       ▼
  Should continue?
       │
  ┌────┴────┐
  │         │
 Yes        No
  │         │
  ▼         ▼
Planner   Reporter ──► Final Report
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Agent Framework | LangGraph | State machine for agentic decision loops |
| LLM | Groq (LLaMA 3.1) | Reasoning and report generation |
| Web Search | Tavily API | Finding relevant research papers |
| PDF Parsing | PyMuPDF | Extracting text from research PDFs |
| Vector Database | ChromaDB | Storing and retrieving paper embeddings |
| Embeddings | sentence-transformers | Offline text embeddings (all-MiniLM-L6-v2) |
| UI | Streamlit | Interactive web interface |
| Observability | LangSmith | Tracing agent decisions |

## ✨ Features

- **Autonomous decision-making** — the agent decides which tools to call and when, based on the research progress so far
- **RAG pipeline** — chunks, embeds, and stores papers for semantic retrieval
- **4 specialized tools** — web search, PDF parsing, knowledge base query, report generation
- **Evaluation harness** — automated quality scoring across multiple test queries
- **Interactive UI** — live status updates and downloadable reports

## 📁 Project Structure

```
research_agent/
├── agent.py          # LangGraph agent brain (planner, tools, reporter)
├── agent_tools.py     # 4 tools: web_search, fetch_pdf, query_kb, generate_report
├── rag.py             # RAG pipeline: chunking, embedding, retrieval
├── eval.py            # Evaluation harness with 5 test cases
├── app.py             # Streamlit web UI
├── config.py          # Centralized configuration
├── requirements.txt   # Python dependencies
└── data/
    ├── pdfs/          # Downloaded research papers
    └── vectordb/      # ChromaDB persistent storage
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Free API keys: [Groq](https://console.groq.com), [Tavily](https://app.tavily.com), [LangSmith](https://smith.langchain.com) (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/research-agent.git
cd research-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Running the Agent

**Run the full agent on a query:**
```bash
python agent.py
```

**Run the evaluation suite:**
```bash
python eval.py full
```

**Launch the web UI:**
```bash
streamlit run app.py
```

## 📊 Evaluation

The agent is evaluated on 5 test queries across 4 metrics:
- **Report length** — does the report contain sufficient detail?
- **Section structure** — does it include Summary, Findings, and References?
- **Topic relevance** — does it cover the key concepts of the query?
- **Output validity** — is the output a valid, non-error report?

## 🔮 Future Improvements

- Add support for multiple LLM providers
- Implement multi-paper comparison reports
- Add citation verification
- Deploy to cloud (Hugging Face Spaces / Render)

## 📄 License

MIT License — feel free to use this project for learning or as a starting point for your own research agent.

---

Built as a learning project to explore agentic AI systems, RAG pipelines, and LLM orchestration.
