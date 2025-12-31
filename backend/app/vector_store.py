import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_vector_store(index_name="multi-agent-rag-gemini"):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Gemini embeddings typically use 768 dimensions
    dimension = 768 
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return PineconeVectorStore(index_name=index_name, embedding=embeddings)