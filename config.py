
"""
config.py
---------
Central config — loads all environment variables in one place.
Switched to Groq API (free, works in India, no quota issues)
"""
import os
from dotenv import load_dotenv
 
load_dotenv()
 
# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY", "")
LANGCHAIN_API_KEY  = os.getenv("LANGCHAIN_API_KEY", "")
 
# ── LangSmith tracing ─────────────────────────────────────────────────────────
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "research-agent")
 
# ── Model settings ────────────────────────────────────────────────────────────
LLM_MODEL       = "llama-3.1-8b-instant"  # free, fast, works in India
EMBEDDING_MODEL = "all-MiniLM-L6-v2"      # free, runs offline on your laptop
 
# ── RAG settings ──────────────────────────────────────────────────────────────
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K_RESULTS   = 5
 
# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
PDF_DIR      = os.path.join(BASE_DIR, "data", "pdfs")
VECTORDB_DIR = os.path.join(BASE_DIR, "data", "vectordb")
 
# ── Validation ────────────────────────────────────────────────────────────────
def check_keys():
    missing = []
    if not GROQ_API_KEY:      missing.append("GROQ_API_KEY")
    if not TAVILY_API_KEY:    missing.append("TAVILY_API_KEY")
    if not LANGCHAIN_API_KEY: missing.append("LANGCHAIN_API_KEY")
    if missing:
        print(f"WARNING: Missing keys in .env: {', '.join(missing)}")
    else:
        print("All API keys loaded successfully.")
 
if __name__ == "__main__":
    check_keys()