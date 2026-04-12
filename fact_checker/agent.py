import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from langchain_pinecone import PineconeVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from tavily import TavilyClient
from langchain_google_vertexai import VertexAIEmbeddings, VectorSearchVectorStore

# Load the .env file
load_dotenv()

# # --- TOOL 1: INTERNAL RAG ---
# def search_internal_knowledge(query: str) -> str:
#     """Searches the company's private internal documents for specific facts."""
#     api_key = os.getenv("PINECONE_API_KEY")
#     embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
#     vector_db = PineconeVectorStore(
#         index_name="gcp-vertex-ai", 
#         embedding=embeddings,
#         pinecone_api_key=api_key
#     )
#     results = vector_db.similarity_search(query, k=3)
#     return "\n---\n".join([res.page_content for res in results])

# # --- INITIALIZE DATABASE ONCE (Global Scope) ---
# _embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

# _vector_db = VectorSearchVectorStore.from_components(
#     project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
#     region=os.getenv("GOOGLE_CLOUD_LOCATION"),
#     index_id=os.getenv("VERTEX_INDEX_ID"),
#     endpoint_id=os.getenv("VERTEX_ENDPOINT_ID"),
#     embedding=_embeddings,
#     gcs_bucket_name=os.getenv("GCS_BUCKET_NAME")  # <--- THE MISSING PIECE!
# )

# # --- TOOL 1: INTERNAL RAG using GCP Vector Search ---
# def search_internal_knowledge(query: str) -> str:
#     """Searches the company's private internal documents for specific facts."""
    
#     # 1. Add error handling so a database timeout doesn't crash the whole agent
#     try:
#         results = _vector_db.similarity_search(query, k=3)
        
#         # 2. Handle empty results gracefully
#         if not results:
#             return "No relevant internal documents found for this query."
        
#         # 3. Include Source Metadata so the AI can cite its sources!
#         formatted_results = []
#         for res in results:
#             # LangChain's GCSLoader usually saves the filename here
#             source = res.metadata.get("source", "Unknown Internal Document") 
#             formatted_results.append(f"Source: {source}\nContent: {res.page_content}")
            
#         return "\n---\n".join(formatted_results)
        
#     except Exception as e:
#         return f"Warning: Could not search internal database due to error: {str(e)}"

# Deploying rag with different logi

# 1. Define a global placeholder, but DO NOT connect yet!
_cached_vector_db = None

def get_vector_db():
    """Lazy initialization: Connects to the database only on the first request."""
    global _cached_vector_db
    
    # If we already connected, just return the existing connection
    if _cached_vector_db is not None:
        return _cached_vector_db
        
    # If this is the first time, build the connection
    _embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    _cached_vector_db = VectorSearchVectorStore.from_components(
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
        region=os.getenv("GOOGLE_CLOUD_LOCATION"),
        index_id=os.getenv("VERTEX_INDEX_ID"),
        endpoint_id=os.getenv("VERTEX_ENDPOINT_ID"),
        embedding=_embeddings,
        gcs_bucket_name=os.getenv("GCS_BUCKET_NAME")
    )
    return _cached_vector_db

# --- 2. INTERNAL RAG TOOL ---
def search_internal_knowledge(query: str) -> str:
    """Searches the company's private internal documents for specific facts."""
    try:
        # Safely grab the database connection
        db = get_vector_db()
        
        results = db.similarity_search(query, k=3)
        if not results:
            return "No relevant internal documents found."
            
        formatted_results = []
        for res in results:
            source = res.metadata.get("source", "Unknown Internal Document") 
            formatted_results.append(f"Source: {source}\nContent: {res.page_content}")
            
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Warning: Database search failed: {str(e)}"

# --- TOOL 2: EXTERNAL WEB SEARCH (TAVILY) ---
def web_search(query: str) -> str:
    """Searches the public internet for up-to-date news, world facts, and public information."""
    api_key = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=api_key)
    
    # Perform a basic search
    response = client.search(query=query, search_depth="basic")
    
    # Format the results cleanly for the LLM
    formatted_results = [f"Source: {res['url']}\nContent: {res['content']}" for res in response.get("results", [])]
    return "\n---\n".join(formatted_results)

# --- DEBUG TOOL: IDENTITY INTERROGATION ---
def check_my_identity(query: str) -> str:
    """Forces the cloud server to reveal its exact IAM service account email."""
    import urllib.request
    try:
        # Pings the internal Google Cloud Metadata server
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
            headers={"Metadata-Flavor": "Google"}
        )
        with urllib.request.urlopen(req) as response:
            return f"My active Service Account is: {response.read().decode('utf-8')}"
    except Exception as e:
        return f"Error fetching identity: {str(e)}"

# --- THE AGENT ---
root_agent = Agent(
    name="FactChecker",
    model="gemini-2.5-flash",
    instruction="""
    You are an advanced dual-engine fact-checker.
    
    RULES:
    1. If the user greets you, respond conversationally. Do NOT use tools.
    2. For world news, general knowledge, or public facts, use the 'web_search' tool.
    3. If a claim requires both, you may use both tools.

    CRITICAL INSTRUCTION: You now have access to a continuous conversation history. 
    1. If the user refers to past statements, previous documents, or uses pronouns (like "it", "that", or "they"), you MUST read the session context to understand what they are referring to.
    2. For company-specific questions, use 'search_internal_knowledge'.
    3. For public facts, use 'web_search'.
    4. If asked about your identity, use 'check_my_identity'.
    
    VERDICT FORMAT:
    VERDICT: [True / False / Unverified]
    REASONING: [Provide explanation]
    """,
    # The API will now happily accept both since they are both custom Python functions!
    tools=[search_internal_knowledge, web_search,check_my_identity] 
)