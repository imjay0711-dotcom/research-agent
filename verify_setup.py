"""
verify_setup.py
---------------
Run this after Step 1 to confirm everything installed correctly.
Usage:  python verify_setup.py
"""

print("=" * 50)
print("  Research Agent - Setup Verification")
print("=" * 50)

checks = [
    ("langgraph",             "LangGraph (agent framework)"),
    ("langchain",             "LangChain (tool utilities)"),
    ("langchain_anthropic",   "LangChain-Anthropic (LLM connector)"),
    ("chromadb",              "ChromaDB (vector database)"),
    ("sentence_transformers", "Sentence-Transformers (embeddings)"),
    ("fitz",                  "PyMuPDF (PDF parser)"),
    ("tavily",                "Tavily (web search)"),
    ("fastapi",               "FastAPI (REST API)"),
    ("streamlit",             "Streamlit (web UI)"),
    ("dotenv",                "python-dotenv (.env loader)"),
    ("pydantic",              "Pydantic (data validation)"),
    ("langsmith",             "LangSmith (observability)"),
]

all_good = True
for module, label in checks:
    try:
        __import__(module)
        print(f"  OK  {label}")
    except ImportError:
        print(f"  MISSING  {label}  <-- run: pip install {module}")
        all_good = False

print()

# Check .env file
import os
from pathlib import Path
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    print("  OK  .env file found")
else:
    print("  WARNING  .env file missing")
    print("           Copy .env.example and rename it to .env")
    print("           Then paste your 3 API keys inside it")

print()
if all_good:
    print("All libraries installed! Ready for Step 2.")
else:
    print("Fix the MISSING items above, then re-run this script.")
print("=" * 50)
