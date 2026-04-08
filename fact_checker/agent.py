import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from langchain_pinecone import PineconeVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from tavily import TavilyClient

# Load the .env file
load_dotenv()

# --- TOOL 1: INTERNAL RAG ---
def search_internal_knowledge(query: str) -> str:
    """Searches the company's private internal documents for specific facts."""
    api_key = os.getenv("PINECONE_API_KEY")
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    vector_db = PineconeVectorStore(
        index_name="gcp-vertex-ai", 
        embedding=embeddings,
        pinecone_api_key=api_key
    )
    results = vector_db.similarity_search(query, k=3)
    return "\n---\n".join([res.page_content for res in results])

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

# --- THE AGENT ---
root_agent = Agent(
    name="FactChecker",
    model="gemini-2.5-flash",
    instruction="""
    You are an advanced dual-engine fact-checker.
    
    RULES:
    1. If the user greets you, respond conversationally. Do NOT use tools.
    2. For company-specific or internal questions, use the 'search_internal_knowledge' tool.
    3. For world news, general knowledge, or public facts, use the 'web_search' tool.
    4. If a claim requires both, you may use both tools.
    
    VERDICT FORMAT:
    VERDICT: [True / False / Unverified]
    REASONING: [Provide explanation]
    """,
    # The API will now happily accept both since they are both custom Python functions!
    tools=[search_internal_knowledge, web_search] 
)