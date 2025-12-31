from langchain_google_genai import ChatGoogleGenerativeAI
from .state import AgentState

# Initialize our "Brain"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def query_rewriter(state: AgentState):
    print("---REWRITING QUERY---")
    messages = state["messages"]
    user_question = messages[-1].content
    
    # Simple but effective production prompt
    prompt = f"Rewrite the following user question to be optimized for a vector database search. Provide only the search query: {user_question}"
    
    response = llm.invoke(prompt)
    
    # We return the update to the state
    return {"queries": [response.content]}