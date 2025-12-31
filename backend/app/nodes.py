from langchain_google_genai import ChatGoogleGenerativeAI
from .state import AgentState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from .vector_store import get_vector_store
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# We use a Pydantic class to force Gemini to give us a structured 'yes' or 'no'
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def grade_documents(state: AgentState):
    print("---CHECKING DOCUMENT RELEVANCE---")
    
    # Set up the grader with structured output
    llm_with_tool = llm.with_structured_output(GradeDocuments)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing relevance of a retrieved document to a user question."),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ])
    
    chain = prompt | llm_with_tool
    
    relevant_docs = []
    for doc in state["documents"]:
        res = chain.invoke({"question": state["queries"][-1], "document": doc})
        if res.binary_score == "yes":
            relevant_docs.append(doc)
            
    # If no docs are relevant, we'll signal the Supervisor to try a different agent
    next_step = "generate" if relevant_docs else "web_search"
    
    return {"documents": relevant_docs, "next_step": next_step}

# Initialize our "Brain"


def query_rewriter(state: AgentState):
    print("---REWRITING QUERY---")
    messages = state["messages"]
    user_question = messages[-1].content
    
    # Simple but effective production prompt
    prompt = f"Rewrite the following user question to be optimized for a vector database search. Provide only the search query: {user_question}"
    
    response = llm.invoke(prompt)
    
    # We return the update to the state
    return {"queries": [response.content]}
def retrieve(state: AgentState):
    print("---RETRIEVING FROM PINECONE---")
    query = state["queries"][-1]
    vector_store = get_vector_store()
    
    # Retrieve top 3 relevant chunks
    docs = vector_store.similarity_search(query, k=3)
    
    return {"documents": [d.page_content for d in docs]}