"""
tools.py
--------
The 4 tools your agent can use to do its work.
Each function is wrapped with @tool so LangGraph can call them automatically.

Tools:
    1. web_search          - searches internet for research papers
    2. fetch_and_parse_pdf - downloads PDF and extracts text
    3. query_knowledge_base- searches ChromaDB for stored content
    4. generate_report     - writes final structured research report
"""

import os
import httpx
import fitz  # PyMuPDF
import chromadb
from langchain.tools import tool
from langchain_groq import ChatGroq
from tavily import TavilyClient
from sentence_transformers import SentenceTransformer
from config import (
    GROQ_API_KEY,
    TAVILY_API_KEY,
    VECTORDB_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    LLM_MODEL,
    PDF_DIR,
)

# ── Setup clients (loaded once when tools.py is imported) ─────────────────────

# Tavily client for web search
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Gemini LLM for report generation

llm = ChatGroq(
    model=LLM_MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=0.3,
)
# Sentence transformer for embeddings (runs offline, no API needed)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# ChromaDB client — stores data in your data/vectordb folder
chroma_client = chromadb.PersistentClient(path=VECTORDB_DIR)
collection = chroma_client.get_or_create_collection(name="research_papers")

# Make sure PDF folder exists
os.makedirs(PDF_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — Web Search
# ─────────────────────────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """
    Search the internet for research papers related to the query.
    Returns a list of paper titles, URLs, and short descriptions.

    Use this tool FIRST when given a new research topic.

    Args:
        query: The research topic to search for. Example: 'transformer attention mechanism'
    """
    try:
        print(f"\n[Tool 1] Searching web for: {query}")

        # Call Tavily API — returns top 5 results
        results = tavily_client.search(
            query=query + " research paper",
            max_results=5,
            search_depth="advanced",
        )

        # Format results into a readable string
        output = f"Web search results for: '{query}'\n"
        output += "=" * 50 + "\n"

        for i, result in enumerate(results["results"], 1):
            output += f"\n{i}. Title: {result['title']}\n"
            output += f"   URL: {result['url']}\n"
            output += f"   Summary: {result['content'][:200]}...\n"

        print(f"[Tool 1] Found {len(results['results'])} results")
        return output

    except Exception as e:
        return f"Web search failed: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — Fetch and Parse PDF
# ─────────────────────────────────────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list:
    """Split text into overlapping chunks of words."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward with overlap

    return chunks


@tool
def fetch_and_parse_pdf(url: str) -> str:
    """
    Download a PDF from a URL, extract its text, and store it in ChromaDB.
    Use this tool after web_search to read the actual content of a paper.

    Args:
        url: The direct URL to a PDF file. Example: 'https://arxiv.org/pdf/1706.03762'
    """
    try:
        print(f"\n[Tool 2] Fetching PDF from: {url}")

        # Download the PDF file
        response = httpx.get(url, follow_redirects=True, timeout=30)

        if response.status_code != 200:
            return f"Failed to download PDF. Status code: {response.status_code}"

        # Save PDF to data/pdfs folder
        pdf_filename = url.split("/")[-1]
        if not pdf_filename.endswith(".pdf"):
            pdf_filename += ".pdf"
        pdf_path = os.path.join(PDF_DIR, pdf_filename)

        with open(pdf_path, "wb") as f:
            f.write(response.content)

        print(f"[Tool 2] PDF saved to: {pdf_path}")

        # Extract text using PyMuPDF
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        if not full_text.strip():
            return "PDF downloaded but no text could be extracted (may be scanned image)."

        print(f"[Tool 2] Extracted {len(full_text)} characters of text")

        # Split text into chunks
        chunks = split_into_chunks(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"[Tool 2] Split into {len(chunks)} chunks")

        # Embed chunks using sentence-transformers
        embeddings = embedding_model.encode(chunks).tolist()

        # Store in ChromaDB
        ids = [f"{pdf_filename}_chunk_{i}" for i in range(len(chunks))]

        # Add to collection (skip if already exists)
        existing = collection.get(ids=ids[:1])
        if not existing["ids"]:
            collection.add(
                embeddings=embeddings,
                documents=chunks,
                ids=ids,
                metadatas=[{"source": url, "filename": pdf_filename}] * len(chunks),
            )
            print(f"[Tool 2] Stored {len(chunks)} chunks in ChromaDB")
        else:
            print(f"[Tool 2] Paper already in ChromaDB, skipping storage")

        # Return first 1000 characters as preview
        preview = full_text[:1000].strip()
        return (
            f"Successfully parsed PDF from {url}\n"
            f"Stored {len(chunks)} chunks in knowledge base.\n\n"
            f"Preview of content:\n{preview}..."
        )

    except Exception as e:
        return f"Failed to fetch/parse PDF: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — Query Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

@tool
def query_knowledge_base(question: str) -> str:
    """
    Search the knowledge base (ChromaDB) for relevant content from stored papers.
    Use this tool after fetch_and_parse_pdf to find specific information.

    Args:
        question: A specific question to search for in stored papers.
                  Example: 'What is the attention mechanism in transformers?'
    """
    try:
        print(f"\n[Tool 3] Querying knowledge base for: {question}")

        # Check if knowledge base has any content
        count = collection.count()
        if count == 0:
            return "Knowledge base is empty. Please fetch some papers first using fetch_and_parse_pdf."

        # Embed the question
        question_embedding = embedding_model.encode([question]).tolist()

        # Search ChromaDB for most similar chunks
        results = collection.query(
            query_embeddings=question_embedding,
            n_results=min(TOP_K_RESULTS, count),
        )

        if not results["documents"][0]:
            return "No relevant content found in knowledge base."

        # Format results
        output = f"Knowledge base results for: '{question}'\n"
        output += "=" * 50 + "\n"

        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0]), 1
        ):
            output += f"\n[Result {i}] Source: {metadata.get('source', 'Unknown')}\n"
            output += f"{doc[:400]}...\n"

        print(f"[Tool 3] Found {len(results['documents'][0])} relevant chunks")
        return output

    except Exception as e:
        return f"Knowledge base query failed: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — Generate Report
# ─────────────────────────────────────────────────────────────────────────────

@tool
def generate_report(research_data: str) -> str:
    """
    Generate a structured research report from collected research data.
    Use this tool LAST after gathering enough information from other tools.

    Args:
        research_data: All the findings collected from web_search and
                       query_knowledge_base combined into one string.
    """
    try:
        print(f"\n[Tool 4] Generating research report...")

        prompt = f"""You are a research assistant. Based on the following research data,
write a clear and structured research report.

RESEARCH DATA:
{research_data}

Write the report in this exact format:

# Research Report

## Summary
(2-3 sentences summarizing the topic)

## Key Findings
(5 bullet points of the most important findings)

## Main Concepts Explained
(Explain the 2-3 most important concepts in simple terms)

## Research Gaps
(What questions are still unanswered?)

## References
(List the paper URLs mentioned in the data)

Keep the report clear, accurate, and easy to understand.
"""
        response = llm.invoke(prompt)
        report = response.content

        print(f"[Tool 4] Report generated successfully ({len(report)} characters)")
        return report

    except Exception as e:
        return f"Report generation failed: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run this file directly to test all tools
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Tool 1 — Web Search")
    print("=" * 50)
    result = web_search.invoke("transformer attention mechanism deep learning")
    print(result)

    print("\n" + "=" * 50)
    print("Tool 1 test complete!")
    print("If you see search results above, Tool 1 is working.")
    print("=" * 50)
