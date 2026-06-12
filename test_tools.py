"""
test_tools.py
-------------
Run this to test all 4 tools one by one.
Usage: python test_tools.py
"""

from agent_tools import web_search, fetch_and_parse_pdf, query_knowledge_base, generate_report

print("\n" + "=" * 60)
print("  Research Agent — Testing All 4 Tools")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# TEST 1 — Web Search
# ─────────────────────────────────────────────────────────────────
print("\n[TEST 1] Web Search...")
print("-" * 40)
result1 = web_search.invoke("transformer attention mechanism deep learning")
print(result1)
print("✅ Tool 1 PASSED" if "results" in result1.lower() or "title" in result1.lower() else "❌ Tool 1 FAILED")

# ─────────────────────────────────────────────────────────────────
# TEST 2 — Fetch and Parse PDF
# ─────────────────────────────────────────────────────────────────
print("\n[TEST 2] Fetch and Parse PDF...")
print("-" * 40)
# Using the original Attention Is All You Need paper from arxiv
pdf_url = "https://arxiv.org/pdf/1706.03762"
result2 = fetch_and_parse_pdf.invoke(pdf_url)
print(result2)
print("✅ Tool 2 PASSED" if "successfully" in result2.lower() else "❌ Tool 2 FAILED")

# ─────────────────────────────────────────────────────────────────
# TEST 3 — Query Knowledge Base
# ─────────────────────────────────────────────────────────────────
print("\n[TEST 3] Query Knowledge Base...")
print("-" * 40)
result3 = query_knowledge_base.invoke("What is the attention mechanism?")
print(result3)
print("✅ Tool 3 PASSED" if "result" in result3.lower() else "❌ Tool 3 FAILED")

# ─────────────────────────────────────────────────────────────────
# TEST 4 — Generate Report
# ─────────────────────────────────────────────────────────────────
print("\n[TEST 4] Generate Report...")
print("-" * 40)
sample_data = f"""
Research on Transformer Attention Mechanism:

Web Search Findings:
- Transformers use self-attention to process sequences
- Introduced in 'Attention Is All You Need' paper (2017)
- Replaced recurrent networks in NLP tasks

Knowledge Base Findings:
{result3[:500]}
"""
result4 = generate_report.invoke(sample_data)
print(result4)
print("✅ Tool 4 PASSED" if "summary" in result4.lower() or "findings" in result4.lower() else "❌ Tool 4 FAILED")

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  All 4 Tools Tested!")
print("  If you see 4 x ✅ PASSED above — Step 2 is COMPLETE!")
print("  Ready to build the Agent Brain in Step 3!")
print("=" * 60)
