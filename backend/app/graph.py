from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import query_rewriter, retrieve, grade_documents

# Initialize the Graph
workflow = StateGraph(AgentState)

# 1. Add our Nodes
workflow.add_node("rewrite_query", query_rewriter)
workflow.add_node("retrieve_docs", retrieve)
workflow.add_node("grade_docs", grade_documents)

# 2. Define the Edges (The Connections)
workflow.set_entry_point("rewrite_query")
workflow.add_edge("rewrite_query", "retrieve_docs")
workflow.add_edge("retrieve_docs", "grade_docs")

# 3. Add Conditional Logic
def decide_to_generate(state):
    if state["next_step"] == "web_search":
        return "web_search"
    return "generate"

# We will define 'web_search' and 'generate' in the next step