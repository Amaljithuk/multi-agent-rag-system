from typing import Annotated, List, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # 'messages' stores the conversation history
    messages: Annotated[list, add_messages]
    
    # 'queries' stores our rewritten search terms
    queries: List[str]
    
    # 'documents' stores the text snippets retrieved from Pinecone
    documents: List[str]
    
    # 'next_step' helps the Supervisor decide who goes next
    next_step: str