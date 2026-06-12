"""
rag.py
------
RAG = Retrieval Augmented Generation

What this file does:
- Takes multiple research papers (PDFs or URLs)
- Splits them into small chunks
- Converts chunks to numbers (embeddings)
- Stores everything in ChromaDB
- Retrieves most relevant chunks when agent asks a question

Think of it like building a smart library:
  - Each paper = a book
  - Each chunk = a page
  - Embeddings = index cards for each page
  - ChromaDB = the library shelf
  - Retrieval = finding the right pages for a question
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

# os — for file and folder operations
import os

# re — regular expressions, used to clean up text
import re

# json — to save and load metadata
import json

# httpx — to download PDFs from the internet
import httpx

# fitz — PyMuPDF library, reads PDF files and extracts text
import fitz

# chromadb — our vector database that stores paper chunks
import chromadb

# SentenceTransformer — converts text to numbers (embeddings)
# runs 100% offline on your laptop, no API needed
from sentence_transformers import SentenceTransformer

# datetime — to record when each paper was ingested
from datetime import datetime

# Import settings from config.py
from config import (
    VECTORDB_DIR,    # path to data/vectordb folder
    PDF_DIR,         # path to data/pdfs folder
    EMBEDDING_MODEL, # "all-MiniLM-L6-v2"
    CHUNK_SIZE,      # 500 words per chunk
    CHUNK_OVERLAP,   # 50 words overlap between chunks
    TOP_K_RESULTS,   # return top 5 results
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SET UP CLIENTS
# ─────────────────────────────────────────────────────────────────────────────

# Load the sentence transformer model
# "all-MiniLM-L6-v2" is a small, fast model that runs on CPU
# It converts any text into a 384-dimension vector (list of 384 numbers)
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model loaded!")

# Connect to ChromaDB
# PersistentClient means data is SAVED to disk (survives restarts)
# Without persistent, data is lost when Python stops
chroma_client = chromadb.PersistentClient(path=VECTORDB_DIR)

# Get or create a collection called "research_papers"
# A collection is like a table in a database
# All our paper chunks go into this one collection
collection = chroma_client.get_or_create_collection(
    name="research_papers",
    # cosine similarity = measure how similar two texts are
    # values closer to 1.0 = more similar
    # values closer to 0.0 = less similar
    metadata={"hnsw:space": "cosine"}
)

# Make sure folders exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTORDB_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw PDF text by removing noise.

    PDFs often have:
    - Extra spaces and newlines
    - Page numbers
    - Strange characters
    - Headers/footers repeated on every page

    This function fixes all of that.

    Input:  raw messy text from PDF
    Output: clean readable text
    """

    # Replace multiple spaces with single space
    # re.sub(pattern, replacement, text)
    text = re.sub(r' +', ' ', text)

    # Replace 3+ newlines with just 2 newlines
    # This removes large blank spaces between sections
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are just numbers (page numbers)
    # ^ = start of line, \d+ = one or more digits, $ = end of line
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)

    # Remove lines shorter than 10 characters (usually noise)
    lines = text.split('\n')
    lines = [line for line in lines if len(line.strip()) > 10]
    text = '\n'.join(lines)

    # Strip leading/trailing whitespace
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TEXT CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split a long text into smaller overlapping chunks.

    WHY chunking?
    - LLMs can only read a limited amount of text at once
    - Smaller chunks = more precise retrieval
    - A chunk of 500 words is easier to search than 10,000 words

    WHY overlap?
    - If a sentence is split between two chunks, overlap ensures
      neither chunk loses important context
    - Example: chunk 1 ends with "...the attention" and chunk 2
      starts with "mechanism works by..." — with overlap, both
      chunks contain the complete phrase

    Example with chunk_size=5, overlap=2:
      Text: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      Chunk 1: [1, 2, 3, 4, 5]
      Chunk 2: [4, 5, 6, 7, 8]  ← starts 2 words back (overlap)
      Chunk 3: [7, 8, 9, 10]

    Input:  long text string
    Output: list of chunk strings
    """

    # Split text into individual words
    words = text.split()

    # If text is shorter than chunk_size, just return it as one chunk
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0  # index of first word in current chunk

    while start < len(words):
        # end = where this chunk ends
        end = start + chunk_size

        # Join words from start to end into a string
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)

        # Move start forward by (chunk_size - overlap)
        # This creates the overlapping effect
        start += chunk_size - overlap

    print(f"  Split into {len(chunks)} chunks "
          f"(chunk_size={chunk_size}, overlap={overlap})")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PDF DOWNLOAD AND TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def download_and_extract_pdf(url: str) -> tuple:
    """
    Download a PDF from a URL and extract its text.

    Steps:
    1. Download PDF using httpx
    2. Save it to data/pdfs/ folder
    3. Open with PyMuPDF (fitz)
    4. Extract text from every page
    5. Clean the text
    6. Return (filename, clean_text)

    Input:  URL string pointing to a PDF
    Output: tuple of (filename, extracted_text)
            or (None, None) if download fails
    """

    try:
        print(f"  Downloading: {url}")

        # Download the PDF
        # follow_redirects=True handles URLs that redirect (like arxiv)
        # timeout=30 means give up after 30 seconds
        response = httpx.get(url, follow_redirects=True, timeout=30)

        # Check if download was successful
        # Status 200 = success, anything else = failure
        if response.status_code != 200:
            print(f"  Download failed! Status: {response.status_code}")
            return None, None

        # Create a filename from the URL
        # Example: "https://arxiv.org/pdf/1706.03762" → "1706.03762.pdf"
        filename = url.split("/")[-1]
        if not filename.endswith(".pdf"):
            filename += ".pdf"

        # Save PDF to data/pdfs/ folder
        pdf_path = os.path.join(PDF_DIR, filename)
        with open(pdf_path, "wb") as f:  # "wb" = write binary
            f.write(response.content)
        print(f"  Saved to: {pdf_path}")

        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)

        # Extract text from every page
        full_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            # get_text() extracts all text from one page
            full_text += page.get_text()

        doc.close()

        # Check if we got any text
        if not full_text.strip():
            print("  Warning: No text extracted (may be a scanned PDF)")
            return filename, None

        print(f"  Extracted {len(full_text):,} characters from {len(doc)} pages")

        # Clean the text
        clean = clean_text(full_text)
        print(f"  After cleaning: {len(clean):,} characters")

        return filename, clean

    except Exception as e:
        print(f"  Error downloading PDF: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — STORE PAPER IN CHROMADB
# ─────────────────────────────────────────────────────────────────────────────

def ingest_paper(url: str, title: str = "Unknown") -> bool:
    """
    Full pipeline: Download PDF → Extract → Chunk → Embed → Store in ChromaDB

    This is the main ingestion function. Call this for each paper you want
    to add to the agent's knowledge base.

    Input:
        url   — URL of the PDF
        title — title of the paper (for metadata)
    Output:
        True  — paper ingested successfully
        False — something went wrong
    """

    print(f"\nIngesting paper: {title}")
    print(f"URL: {url}")
    print("-" * 40)

    # ── Check if already ingested ─────────────────────────────────────────────
    # Use the URL as a unique ID to check if paper already exists
    existing = collection.get(
        where={"source_url": url}  # search by metadata field
    )
    if existing["ids"]:
        print(f"  Paper already in knowledge base — skipping")
        return True

    # ── Download and extract ──────────────────────────────────────────────────
    filename, text = download_and_extract_pdf(url)

    if not text:
        print("  Failed to extract text — skipping paper")
        return False

    # ── Split into chunks ─────────────────────────────────────────────────────
    chunks = split_into_chunks(text)

    # ── Create embeddings ─────────────────────────────────────────────────────
    print(f"  Creating embeddings for {len(chunks)} chunks...")

    # encode() converts each chunk (string) into a vector (list of 384 numbers)
    # The model finds the "meaning" of each chunk and represents it as numbers
    # Chunks with similar meaning will have similar vectors
    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=True,  # shows a progress bar while encoding
        batch_size=32,            # process 32 chunks at a time
    ).tolist()  # convert numpy array to Python list (ChromaDB needs this)

    print(f"  Created {len(embeddings)} embeddings")

    # ── Store in ChromaDB ─────────────────────────────────────────────────────
    # Each chunk needs a unique ID
    # Format: "filename_chunk_0", "filename_chunk_1", etc.
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

    # Metadata = extra information stored alongside each chunk
    # We can later filter or display this information
    metadatas = []
    for i in range(len(chunks)):
        metadatas.append({
            "source_url": url,           # where the paper came from
            "filename": filename,         # local filename
            "title": title,               # paper title
            "chunk_index": i,             # position of this chunk in the paper
            "total_chunks": len(chunks),  # total number of chunks
            "ingested_at": datetime.now().isoformat(),  # when it was added
        })

    # Add everything to ChromaDB
    # ChromaDB stores: id + embedding + document text + metadata
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,      # the actual text of each chunk
        metadatas=metadatas,   # extra info about each chunk
    )

    print(f"  Stored {len(chunks)} chunks in ChromaDB")
    print(f"  Paper '{title}' ingested successfully!")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — RETRIEVE RELEVANT CHUNKS
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K_RESULTS,
             filter_title: str = None) -> list:
    """
    Find the most relevant chunks for a given query.

    How it works:
    1. Convert query text to embedding (vector)
    2. Compare query vector with all stored chunk vectors
    3. Return chunks whose vectors are most similar to query vector
    4. "Most similar" = same topic/meaning

    Example:
        query = "how does attention mechanism work?"
        → finds chunks from papers that explain attention

    Input:
        query        — the question to search for
        top_k        — how many chunks to return (default 5)
        filter_title — optional: only search within a specific paper
    Output:
        list of dicts, each containing:
          - text     : the chunk text
          - source   : URL of the paper
          - title    : paper title
          - score    : similarity score (higher = more relevant)
          - chunk_id : position in the paper
    """

    # Check if knowledge base has any content
    total = collection.count()
    if total == 0:
        print("Knowledge base is empty! Please ingest papers first.")
        return []

    print(f"\nSearching knowledge base for: '{query}'")
    print(f"Total chunks in database: {total}")

    # Convert query to embedding
    # Same model that encoded the chunks — ensures compatible vectors
    query_embedding = embedding_model.encode([query]).tolist()

    # Build optional filter
    # If filter_title is provided, only search within that paper
    where_filter = None
    if filter_title:
        where_filter = {"title": filter_title}

    # Search ChromaDB
    # query_embeddings — the vector we're searching for
    # n_results — how many results to return
    # include — what data to return with each result
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, total),  # can't return more than total
        include=["documents", "metadatas", "distances"],
        where=where_filter,
    )

    # Format results into a clean list of dicts
    formatted = []
    for i in range(len(results["documents"][0])):
        formatted.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source_url", ""),
            "title": results["metadatas"][0][i].get("title", "Unknown"),
            "score": 1 - results["distances"][0][i],  # convert distance to score
            "chunk_id": results["metadatas"][0][i].get("chunk_index", 0),
        })

    print(f"Found {len(formatted)} relevant chunks")
    return formatted


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — KNOWLEDGE BASE STATS
# ─────────────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """
    Return statistics about the knowledge base.
    Useful for checking what papers are stored.
    """
    total_chunks = collection.count()

    # Get all metadata to find unique papers
    if total_chunks == 0:
        return {"total_chunks": 0, "papers": [], "total_papers": 0}

    all_data = collection.get(include=["metadatas"])
    # Find unique paper titles
    papers = list(set([
        m.get("title", "Unknown")
        for m in all_data["metadatas"]
    ]))

    return {
        "total_chunks": total_chunks,
        "total_papers": len(papers),
        "papers": papers,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST — Run this file directly to test the RAG pipeline
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Testing RAG Pipeline")
    print("=" * 60)

    # List of papers to ingest
    # These are famous AI papers from arxiv (free to access)
    papers = [
        {
            "url": "https://arxiv.org/pdf/1706.03762",
            "title": "Attention Is All You Need"
        },
        {
            "url": "https://arxiv.org/pdf/1810.04805",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers"
        },
    ]

    # Ingest all papers
    print("\n--- INGESTING PAPERS ---")
    for paper in papers:
        ingest_paper(paper["url"], paper["title"])

    # Check stats
    print("\n--- KNOWLEDGE BASE STATS ---")
    stats = get_stats()
    print(f"Total papers : {stats['total_papers']}")
    print(f"Total chunks : {stats['total_chunks']}")
    print(f"Papers stored: {stats['papers']}")

    # Test retrieval
    print("\n--- TESTING RETRIEVAL ---")
    test_queries = [
        "What is the attention mechanism?",
        "How does BERT work?",
        "What is multi-head attention?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retrieve(query, top_k=2)
        for i, r in enumerate(results, 1):
            print(f"  Result {i}: [{r['title']}] score={r['score']:.3f}")
            print(f"  Text: {r['text'][:150]}...")

    print("\n" + "=" * 60)
    print("RAG Pipeline test complete!")
    print("If you see results above — Step 4 is COMPLETE!")
    print("=" * 60)
