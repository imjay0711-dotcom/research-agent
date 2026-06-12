"""
agent.py
--------
The BRAIN of your Research Agent.
This is a LangGraph state machine that decides:
  - Which tool to call next
  - When to stop searching
  - When to generate the final report

How it works:
  User Query
      ↓
  [Planner Node]  ← LLM decides what to do next
      ↓
  [Tool Node]     ← Actually runs the tool
      ↓
  Should I continue? ← Checks if we have enough info
      ↓ Yes               ↓ No
  [Planner Node]    [Reporter Node] ← Writes final report
                         ↓
                    Final Report
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

# os — used to set environment variables
import os

# json — used to parse tool call arguments from LLM response
import json

# typing — used to define types for our state variables
from typing import TypedDict, Annotated, List

# operator.add — used to combine lists when updating state
import operator

# HumanMessage, AIMessage, SystemMessage — different types of chat messages
# that we send to and receive from the LLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ChatGroq — the Groq LLM client we use for reasoning
from langchain_groq import ChatGroq

# StateGraph — the main LangGraph class that builds our decision graph
# END — a special marker that tells LangGraph "stop here, we are done"
from langgraph.graph import StateGraph, END

# ToolNode — a special LangGraph node that automatically runs tools
from langgraph.prebuilt import ToolNode

# Import our 4 tools from agent_tools.py
from agent_tools import (
    web_search,
    fetch_and_parse_pdf,
    query_knowledge_base,
    generate_report,
)

# Import settings from config.py
from config import GROQ_API_KEY, LLM_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DEFINE THE STATE
# ─────────────────────────────────────────────────────────────────────────────

# AgentState is like a "memory box" that travels through the entire agent.
# Every node (planner, tool, reporter) can READ from it and WRITE to it.
# Think of it as the agent's working memory.

class AgentState(TypedDict):
    # messages — the full conversation history between user and agent
    # Annotated[List, operator.add] means: when we add new messages,
    # append them to the existing list (don't replace the whole list)
    messages: Annotated[List, operator.add]

    # query — the original research question the user asked
    # We keep this so the agent never forgets what it was trying to answer
    query: str

    # tool_results — a list of all results from every tool call so far
    # Each time a tool runs, its output gets added here
    tool_results: Annotated[List, operator.add]

    # final_report — the finished research report
    # This starts as empty string and gets filled in the reporter node
    final_report: str

    # step_count — counts how many tool calls have been made
    # We use this to stop the agent after 10 steps (prevents infinite loops)
    step_count: int


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — SET UP THE LLM WITH TOOLS
# ─────────────────────────────────────────────────────────────────────────────

# Create list of all tools the agent can use
# The agent will choose FROM this list automatically
tools = [web_search, fetch_and_parse_pdf, query_knowledge_base, generate_report]

# Create the Groq LLM
llm = ChatGroq(
    model=LLM_MODEL,           # which AI model to use (llama-3.1-8b-instant)
    groq_api_key=GROQ_API_KEY, # our API key
    temperature=0,             # 0 = more focused/deterministic answers
                               # 1 = more creative/random answers
)

# bind_tools() tells the LLM about our 4 tools
# Now the LLM can say "I want to call web_search with query=X"
# instead of just returning text
llm_with_tools = llm.bind_tools(tools)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DEFINE NODE 1: PLANNER
# ─────────────────────────────────────────────────────────────────────────────

# The planner node is the "thinking" part of the agent.
# It looks at what has happened so far and decides:
# "What should I do next? Which tool should I call?"

def planner_node(state: AgentState) -> dict:
    """
    The Planner reads the current state and decides which tool to call next.

    Input:  current state (messages, query, tool_results so far)
    Output: updated state with new message (either a tool call or final answer)
    """

    print(f"\n{'='*50}")
    print(f"[Planner] Step {state['step_count'] + 1} — Thinking...")
    print(f"[Planner] Query: {state['query']}")
    print(f"[Planner] Tools used so far: {state['step_count']}")

    # Build the system prompt — this tells the LLM its role and strategy
    system_prompt = """You are an expert research agent. Your job is to research a topic thoroughly.

You have access to these tools:
1. web_search — search the internet for research papers (USE THIS FIRST)
2. fetch_and_parse_pdf — download and read a PDF paper from a URL
3. query_knowledge_base — search stored papers for specific information
4. generate_report — write the final research report (USE THIS LAST)

YOUR RESEARCH STRATEGY:
Step 1: Always start with web_search to find relevant papers
Step 2: Use fetch_and_parse_pdf on 1-2 promising paper URLs from search results
Step 3: Use query_knowledge_base to find specific details about the topic
Step 4: Use generate_report ONLY when you have enough information (after steps 1-3)

IMPORTANT RULES:
- Call only ONE tool at a time
- After web_search, always fetch at least one PDF
- After fetching PDF, always query the knowledge base
- After querying knowledge base, generate the report
- Do NOT repeat the same tool with the same input
"""

    # Combine system prompt + full conversation history
    # This gives the LLM complete context of everything that happened
    all_messages = [SystemMessage(content=system_prompt)] + state["messages"]

    # Call the LLM — it will either:
    # 1. Return a tool_call (saying "call web_search with X")
    # 2. Return a text response (if it thinks it's done)
    response = llm_with_tools.invoke(all_messages)

    print(f"[Planner] Decision: {response.content if response.content else 'Calling a tool...'}")

    # Return updated state — add the LLM's response to messages
    return {
        "messages": [response],           # add LLM response to history
        "step_count": state["step_count"] + 1,  # increment step counter
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — DEFINE NODE 2: TOOL EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

# ToolNode is a pre-built LangGraph node that:
# 1. Reads the tool_call from the last LLM message
# 2. Actually runs that tool (web_search, fetch_pdf, etc.)
# 3. Returns the tool result as a message
# We don't need to write this ourselves — LangGraph handles it!

tool_node = ToolNode(tools)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — DEFINE NODE 3: REPORTER
# ─────────────────────────────────────────────────────────────────────────────

# The reporter node runs at the very end.
# It collects ALL tool results gathered so far and
# calls generate_report to write the final structured report.

def reporter_node(state: AgentState) -> dict:
    """
    The Reporter collects all research findings and generates the final report.

    Input:  state with all tool_results collected so far
    Output: state with final_report filled in
    """

    print(f"\n{'='*50}")
    print("[Reporter] Generating final research report...")

    # Collect all text from the conversation history
    # Look through all messages and find tool results
    all_findings = f"Research Query: {state['query']}\n\n"
    all_findings += "=" * 50 + "\n"
    all_findings += "COLLECTED RESEARCH FINDINGS:\n"
    all_findings += "=" * 50 + "\n\n"

    # Go through every message in history
    for msg in state["messages"]:
        # If it's an AI message with content (text response)
        if isinstance(msg, AIMessage) and msg.content:
            all_findings += f"{msg.content}\n\n"

        # If it's a tool result message
        elif hasattr(msg, 'content') and hasattr(msg, 'name'):
            all_findings += f"[{msg.name} result]:\n{msg.content}\n\n"

    # Call generate_report tool with all findings
    report = generate_report.invoke(all_findings)

    print(f"[Reporter] Report generated! ({len(report)} characters)")

    # Return updated state with the final report
    return {
        "final_report": report,
        "messages": [AIMessage(content=f"Research complete! Here is your report:\n\n{report}")]
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — DEFINE THE ROUTING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

# This function is called after EVERY tool execution.
# It looks at the current state and decides:
# "Should I go back to planner for more research?"
# "Or should I go to reporter to write the final report?"

def should_continue(state: AgentState) -> str:
    """
    Decides what to do after each tool call.

    Returns:
        "continue" → go back to planner for more tool calls
        "end"      → go to reporter to write final report
    """

    messages = state["messages"]
    last_message = messages[-1]  # get the most recent message

    # Safety check — if we've done 8+ steps, force end to prevent infinite loop
    if state["step_count"] >= 8:
        print("[Router] Max steps reached — forcing report generation")
        return "end"

    # Check if the last message has tool_calls
    # If yes → the LLM wants to call another tool → continue
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        print(f"[Router] LLM wants to call: {tool_name}")

        # If LLM wants to generate_report → we go to reporter node instead
        if tool_name == "generate_report":
            print("[Router] Report requested → going to reporter")
            return "end"

        # Otherwise continue with more tool calls
        print("[Router] Continuing research...")
        return "continue"

    # If no tool calls → LLM gave a text response → end
    print("[Router] No more tool calls → ending")
    return "end"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

# Now we connect all the nodes together into a graph.
# Think of it like drawing a flowchart with arrows between boxes.

def build_agent():
    """
    Builds and compiles the LangGraph agent.
    Returns a runnable agent graph.
    """

    # Create a new StateGraph using our AgentState as the memory structure
    graph = StateGraph(AgentState)

    # ADD NODES — each node is a "box" in our flowchart
    graph.add_node("planner", planner_node)   # box 1: thinks and decides
    graph.add_node("tools", tool_node)         # box 2: runs the chosen tool
    graph.add_node("reporter", reporter_node)  # box 3: writes final report

    # SET ENTRY POINT — where does the graph start?
    # Always start at planner
    graph.set_entry_point("planner")

    # ADD EDGES — arrows between boxes

    # Arrow 1: planner → tools (after planner decides, run the tool)
    graph.add_edge("planner", "tools")

    # Arrow 2: tools → ??? (after tool runs, where do we go?)
    # This is a CONDITIONAL edge — it calls should_continue() to decide
    # If should_continue returns "continue" → go back to planner
    # If should_continue returns "end" → go to reporter
    graph.add_conditional_edges(
        "tools",           # FROM this node
        should_continue,   # CALL this function to decide
        {
            "continue": "planner",   # if "continue" → go to planner
            "end": "reporter",       # if "end" → go to reporter
        }
    )

    # Arrow 3: reporter → END (after report is written, we are done)
    graph.add_edge("reporter", END)

    # COMPILE — turn the graph definition into a runnable agent
    agent = graph.compile()

    print("Agent brain built successfully!")
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — RUN THE AGENT
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(query: str) -> str:
    """
    Run the research agent with a user query.

    Input:  query — the research topic (e.g. "transformer attention mechanism")
    Output: final_report — the complete research report as a string
    """

    print(f"\n{'#'*60}")
    print(f"  RESEARCH AGENT STARTING")
    print(f"  Query: {query}")
    print(f"{'#'*60}")

    # Build the agent graph
    agent = build_agent()

    # Create the initial state — this is the starting memory of the agent
    initial_state = {
        "messages": [HumanMessage(content=f"Research this topic thoroughly: {query}")],
        "query": query,          # save the original query
        "tool_results": [],      # empty list — no results yet
        "final_report": "",      # empty — no report yet
        "step_count": 0,         # zero steps taken so far
    }

    # RUN THE AGENT
    # invoke() starts the graph and runs until it reaches END
    # The agent will loop through planner → tools → planner → tools...
    # until should_continue() returns "end", then runs reporter
    final_state = agent.invoke(initial_state)

    # Return the final report from the completed state
    return final_state["final_report"]


# ─────────────────────────────────────────────────────────────────────────────
# TEST — Run this file directly to test the agent
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test query — feel free to change this!
    test_query = "transformer attention mechanism in deep learning"

    print("Starting Research Agent test...")
    report = run_agent(test_query)

    print(f"\n{'#'*60}")
    print("FINAL RESEARCH REPORT:")
    print(f"{'#'*60}")
    print(report)

    print(f"\n{'#'*60}")
    print("Agent test complete!")
    print("If you see a report above — Step 3 is COMPLETE!")
    print(f"{'#'*60}")
