
# 🧠 Enterprise RAG Architecture: Vertex AI Vector Search

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Vertex_AI-4285F4.svg)
![Vector Search](https://img.shields.io/badge/Vector_DB-Vertex_Native-4285F4.svg)
![Tavily](https://img.shields.io/badge/Search-Tavily_AI-green.svg)

An enterprise-grade, serverless AI agent deployed on **Google Cloud Vertex AI Reasoning Engines**. 

This iteration upgrades the architecture from a managed SaaS vector database (Pinecone) to **Google Cloud's native Vertex AI Vector Search**. This ensures a 100% Google-native data pipeline where sensitive internal company documents never leave the GCP security perimeter.

---

## 🏗️ The Decoupled Architecture

Unlike beginner-friendly databases, Vertex AI utilizes a **Decoupled Architecture** for infinite global scaling and zero-downtime updates:
1.  **The Index (The Math):** The raw 768-dimension vectors and Google's highly optimized ScaNN (Tree-AH) sorting algorithm.
2.  **The Index Endpoint (The Server):** The dedicated, load-balanced compute hardware that loads the Index into RAM and serves API requests.

---

## 🚀 GCP Infrastructure Setup

Before running any code, the vector database must be provisioned in the Google Cloud Console.

### 1. Create the Index
Navigate to **Vertex AI** -> **Vector Search** -> **Indexes** and create a new index with these strict parameters:
* **Dimensions:** `768` (Must match `text-embedding-004`)
* **Update Method:** `Stream` (Enables real-time data injection)
* **Distance Measure:** `Dot product distance` *(See Troubleshooting below)*
* **GCS folder URI:** A path to an empty cloud storage folder for periodic backups.

### 2. Create the Endpoint & Deploy
1.  Navigate to **Index Endpoints** and create a **Public Endpoint**.
2.  Click **Deploy Index** on your new endpoint and attach the index created in Step 1.
3.  *Note: Hardware provisioning takes 15–30 minutes.*

---

## 🛠️ Environment Configuration (`.env`)

Add your new Vertex AI resource IDs to your environment file. **Never commit this to version control.**

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GCS_BUCKET_NAME=your-bucket-name

# Vertex AI Vector Search
VERTEX_INDEX_ID=projects/123456789/locations/us-central1/indexes/987654321
VERTEX_ENDPOINT_ID=projects/123456789/locations/us-central1/indexEndpoints/11223344

# External APIs
TAVILY_API_KEY=tvly-your-key-here
```

---

## 🗄️ Phase 1: Native Data Ingestion

This script reads internal documents from Cloud Storage, chunks them, generates embeddings, and streams them directly into the Vertex AI Endpoint.

**File: `ingest_data.py`**
```python
import os
from dotenv import load_dotenv
from langchain_google_community import GCSDirectoryLoader
from langchain_google_vertexai import VertexAIEmbeddings, VectorSearchVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

print("🔍 Reading docs from GCS...")
loader = GCSDirectoryLoader(
    project_name=os.getenv("GOOGLE_CLOUD_PROJECT"), 
    bucket=os.getenv("GCS_BUCKET_NAME")
)
docs = loader.load()

print("✂️ Chunking...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

print("🧠 Connecting to Vertex AI Vector Search...")
# LangChain requires .from_components() for existing enterprise endpoints
vector_db = VectorSearchVectorStore.from_components(
    project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
    region=os.getenv("GOOGLE_CLOUD_LOCATION"),
    index_id=os.getenv("VERTEX_INDEX_ID"),
    endpoint_id=os.getenv("VERTEX_ENDPOINT_ID"),
    embedding=VertexAIEmbeddings(model_name="text-embedding-004"),
    stream_update=True,
    gcs_bucket_name=os.getenv("GCS_BUCKET_NAME") # Required staging bucket
)

print("📤 Streaming chunks to Endpoint...")
vector_db.add_documents(chunks)
print("✅ Data successfully ingested into Vertex AI!")
```

---

## 🤖 Phase 2: The Optimized Hybrid Agent

The agent is designed with a globally initialized database connection to eliminate cold-start latency during tool execution. It also extracts document metadata to provide accurate citations.

**File: `agent.py`**
```python
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from langchain_google_vertexai import VertexAIEmbeddings, VectorSearchVectorStore
from tavily import TavilyClient

load_dotenv()

# --- 1. GLOBAL DATABASE INITIALIZATION ---
# Connect once at server boot to prevent latency during tool calls
_vector_db = VectorSearchVectorStore.from_components(
    project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
    region=os.getenv("GOOGLE_CLOUD_LOCATION"),
    index_id=os.getenv("VERTEX_INDEX_ID"),
    endpoint_id=os.getenv("VERTEX_ENDPOINT_ID"),
    embedding=VertexAIEmbeddings(model_name="text-embedding-004"),
    gcs_bucket_name=os.getenv("GCS_BUCKET_NAME")
)

# --- 2. INTERNAL RAG TOOL ---
def search_internal_knowledge(query: str) -> str:
    """Searches the company's private internal documents for specific facts."""
    try:
        results = _vector_db.similarity_search(query, k=3)
        if not results:
            return "No relevant internal documents found."
            
        # Extract metadata to allow the LLM to cite its sources
        formatted_results = []
        for res in results:
            source = res.metadata.get("source", "Unknown Internal Document") 
            formatted_results.append(f"Source: {source}\nContent: {res.page_content}")
            
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Warning: Database search failed: {str(e)}"

# --- 3. EXTERNAL WEB TOOL ---
def web_search(query: str) -> str:
    """Searches the public internet for up-to-date news and global facts."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query=query, search_depth="basic")
    formatted_results = [f"Source: {res['url']}\nContent: {res['content']}" for res in response.get("results", [])]
    return "\n---\n".join(formatted_results)

# --- 4. THE AGENT ---
root_agent = Agent(
    name="EnterpriseFactChecker",
    model="gemini-2.5-flash",
    instruction="""
    You are an advanced dual-engine internal fact-checker.
    1. If the user greets you, respond conversationally. Do NOT use tools.
    2. For company-specific or internal questions, use 'search_internal_knowledge'.
    3. For world news or public facts, use 'web_search'.
    
    VERDICT FORMAT:
    VERDICT: [True / False / Unverified]
    REASONING: [Provide explanation, citing the Source document if applicable]
    """,
    tools=[search_internal_knowledge, web_search] 
)
```

---

## 🚨 Troubleshooting & Common Gotchas

### 1. `NotImplementedError` when initializing VectorStore
* **The Cause:** Attempting to use `.from_documents()` or `VectorSearchVectorStore(...)` directly. LangChain treats GCP enterprise endpoints differently than SaaS databases.
* **The Fix:** You must initialize the connection first using `.from_components(...)` and then separately call `.add_documents()`.

### 2. `ValueError: gcs_bucket_name is required for api_version='v1'`
* **The Cause:** Even when utilizing `stream_update=True`, the LangChain Vertex AI wrapper fundamentally requires a Cloud Storage bucket to act as a temporary staging/formatting ground for the API requests.
* **The Fix:** Pass `gcs_bucket_name=os.getenv("GCS_BUCKET_NAME")` into the `.from_components()` initialization.

### 3. Why `Dot Product` instead of `Cosine`?
* **The Cause:** Google Cloud strongly warns against using Cosine distance for `text-embedding-004`.
* **The Fix:** Always use **Dot Product**. Google's embeddings are natively *unit-normalized* (length of 1). For normalized vectors, Dot Product and Cosine Similarity are mathematically identical, but Dot Product skips the division step, making your enterprise searches significantly faster and cheaper.

### 4. `last_update_time` Stale Session Error (Local Testing)
* **The Cause:** If the local Uvicorn server (`adk web`) is abruptly killed (e.g., stopping an infinite loop), the local SQLite database becomes corrupted.
* **The Fix:** Stop the server, delete the hidden `.adk` folder in your project directory, and restart `adk web`. The server will safely rebuild a clean local memory database.
```