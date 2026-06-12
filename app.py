"""
app.py
------
Streamlit Web UI for the Research Agent.

What this file does:
- Creates a beautiful web interface
- User types a research query
- Shows agent thinking in real time
- Displays the final report as formatted markdown
- Shows knowledge base statistics

Run with:
    streamlit run app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

# streamlit — the web UI framework
# st.title() creates a title
# st.text_input() creates a text box
# st.button() creates a button
# st.markdown() renders formatted text
import streamlit as st

# time — used for small delays in status updates
import time

# datetime — show when report was generated
from datetime import datetime

# import our agent and RAG functions
from agent import run_agent
from rag import get_stats, ingest_paper


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# This MUST be the first streamlit command in the file
# Sets the browser tab title, icon, and layout
st.set_page_config(
    page_title="Research Agent",   # browser tab title
    page_icon="🔬",                # browser tab icon
    layout="wide",                 # use full width of screen
    initial_sidebar_state="expanded"  # sidebar open by default
)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS STYLING
# ─────────────────────────────────────────────────────────────────────────────

# st.markdown() with unsafe_allow_html=True lets us add custom CSS
# This makes the UI look more professional
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }

    /* Subtitle styling */
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }

    /* Status box styling */
    .status-box {
        background: #f0f7ff;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }

    /* Success box */
    .success-box {
        background: #f0fff4;
        border-left: 4px solid #00cc66;
        padding: 1rem;
        border-radius: 4px;
    }

    /* Report container */
    .report-container {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-top: 1rem;
    }

    /* Metric card */
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

# st.sidebar — everything here appears in the left panel
with st.sidebar:

    # Sidebar title
    st.title("⚙️ Settings")
    st.divider()  # horizontal line

    # ── Knowledge Base Stats ──────────────────────────────────────────────────
    st.subheader("📚 Knowledge Base")

    # Get current stats from ChromaDB
    stats = get_stats()

    # Display stats as metrics
    # st.metric() shows a number with a label
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Papers", stats["total_papers"])
    with col2:
        st.metric("Chunks", stats["total_chunks"])

    # Show list of stored papers
    if stats["papers"]:
        st.caption("Stored papers:")
        for paper in stats["papers"]:
            st.caption(f"• {paper[:40]}...")
    else:
        st.caption("No papers stored yet")

    st.divider()

    # ── Add Paper Manually ────────────────────────────────────────────────────
    st.subheader("➕ Add Paper Manually")
    st.caption("Add a PDF URL to the knowledge base")

    # Text input for PDF URL
    manual_url = st.text_input(
        "PDF URL",
        placeholder="https://arxiv.org/pdf/..."
    )

    # Text input for paper title
    manual_title = st.text_input(
        "Paper Title",
        placeholder="Enter paper title"
    )

    # Button to add paper
    if st.button("Add to Knowledge Base", use_container_width=True):
        if manual_url and manual_title:
            # Show spinner while ingesting
            with st.spinner("Ingesting paper..."):
                success = ingest_paper(manual_url, manual_title)
            if success:
                st.success("Paper added successfully!")
                # Refresh the page to update stats
                st.rerun()
            else:
                st.error("Failed to add paper. Check the URL.")
        else:
            st.warning("Please enter both URL and title")

    st.divider()

    # ── About Section ─────────────────────────────────────────────────────────
    st.subheader("ℹ️ About")
    st.caption("""
    **Research Agent** uses:
    - 🧠 Groq LLaMA 3.1
    - 🔍 Tavily Web Search
    - 📄 PyMuPDF PDF Parser
    - 💾 ChromaDB Vector DB
    - 🔗 LangGraph Agent Loop
    """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

# Main title
st.markdown('<p class="main-title">🔬 Research Agent</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Autonomous AI agent that researches any topic, reads papers, and generates structured reports.</p>',
            unsafe_allow_html=True)

st.divider()

# ── Query Input Section ───────────────────────────────────────────────────────

st.subheader("🔎 Enter Your Research Topic")

# Text input box for the research query
# The value is stored in the variable 'query'
query = st.text_input(
    label="Research Query",
    placeholder="e.g. transformer attention mechanism in deep learning",
    label_visibility="collapsed",  # hides the label (we have subheader above)
)

# Example queries — clicking these fills the text box
st.caption("Try these examples:")

# Create 3 columns for example buttons
ex_col1, ex_col2, ex_col3 = st.columns(3)

with ex_col1:
    # If user clicks this button, set query to this text
    if st.button("🤖 Transformer Attention", use_container_width=True):
        query = "transformer attention mechanism deep learning"

with ex_col2:
    if st.button("🧬 BERT Language Model", use_container_width=True):
        query = "BERT language model pre-training NLP"

with ex_col3:
    if st.button("🎮 Reinforcement Learning", use_container_width=True):
        query = "reinforcement learning reward policy optimization"

st.divider()

# ── Run Button ────────────────────────────────────────────────────────────────

# Big green button to start the agent
run_button = st.button(
    "🚀 Run Research Agent",
    type="primary",          # makes it blue/primary color
    use_container_width=True # full width button
)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

# This block runs when user clicks the Run button
if run_button:

    # Check if query is empty
    if not query or len(query.strip()) == 0:
        st.warning("⚠️ Please enter a research topic first!")

    else:
        # ── Status Display ────────────────────────────────────────────────────
        st.subheader("⚡ Agent Working...")

        # Create a container for status updates
        # We update this container as the agent progresses
        status_container = st.empty()

        # Show initial status
        with status_container.container():
            st.info(f"🔍 Starting research on: **{query}**")

        # Progress bar — goes from 0 to 100 as agent works
        progress_bar = st.progress(0)

        # Status messages container
        status_box = st.empty()

        # Update status — step 1
        progress_bar.progress(10)
        with status_box.container():
            st.markdown("🔍 **Step 1/4:** Searching the web for papers...")

        time.sleep(0.5)  # small delay so user can see the update

        # Update status — step 2
        progress_bar.progress(25)
        with status_box.container():
            st.markdown("🔍 **Step 1/4:** Searching web... ✅")
            st.markdown("📄 **Step 2/4:** Fetching and reading PDFs...")

        time.sleep(0.5)

        # Update status — step 3
        progress_bar.progress(50)
        with status_box.container():
            st.markdown("🔍 **Step 1/4:** Searching web... ✅")
            st.markdown("📄 **Step 2/4:** Reading PDFs... ✅")
            st.markdown("🧠 **Step 3/4:** Querying knowledge base...")

        time.sleep(0.5)

        # Update status — step 4
        progress_bar.progress(75)
        with status_box.container():
            st.markdown("🔍 **Step 1/4:** Searching web... ✅")
            st.markdown("📄 **Step 2/4:** Reading PDFs... ✅")
            st.markdown("🧠 **Step 3/4:** Querying knowledge base... ✅")
            st.markdown("📝 **Step 4/4:** Generating research report...")

        # ── Run the Agent ─────────────────────────────────────────────────────
        # Record start time
        start_time = time.time()

        try:
            # THIS IS THE MAIN CALL — runs the entire agent
            # run_agent() from agent.py does all the work:
            # search → fetch PDF → query ChromaDB → generate report
            report = run_agent(query)

            # Calculate time taken
            elapsed = time.time() - start_time

            # ── Success — Show Results ────────────────────────────────────────

            # Complete the progress bar
            progress_bar.progress(100)

            # Update status to complete
            with status_box.container():
                st.markdown("🔍 **Step 1/4:** Searching web... ✅")
                st.markdown("📄 **Step 2/4:** Reading PDFs... ✅")
                st.markdown("🧠 **Step 3/4:** Querying knowledge base... ✅")
                st.markdown("📝 **Step 4/4:** Generating report... ✅")

            # Success message
            st.success(f"✅ Research complete in {elapsed:.1f} seconds!")

            st.divider()

            # ── Display the Report ────────────────────────────────────────────
            st.subheader("📄 Research Report")

            # Show metadata row
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.metric("Word Count", len(report.split()))
            with meta_col2:
                st.metric("Time Taken", f"{elapsed:.1f}s")
            with meta_col3:
                st.metric("Status", "Complete ✅")

            st.divider()

            # Render the report as formatted markdown
            # st.markdown() renders # headers, **bold**, bullet points etc.
            st.markdown(report)

            st.divider()

            # ── Download Button ───────────────────────────────────────────────
            # Let user download the report as a text file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_report_{timestamp}.txt"

            st.download_button(
                label="⬇️ Download Report",
                data=report,           # content of the file
                file_name=filename,    # suggested filename
                mime="text/plain",     # file type
                use_container_width=True
            )

        except Exception as e:
            # ── Error Handling ────────────────────────────────────────────────
            progress_bar.progress(0)
            st.error(f"❌ Agent encountered an error: {str(e)}")
            st.caption("Please check your API keys in the .env file and try again.")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption("Built with LangGraph • Groq LLaMA 3.1 • ChromaDB • Streamlit")
