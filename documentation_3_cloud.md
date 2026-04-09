# 🧠 Enterprise RAG: Pinecone to Vertex AI Migration Guide

This document outlines the architecture, deployment strategy, and troubleshooting protocols for migrating a Retrieval-Augmented Generation (RAG) agent from a managed SaaS database (Pinecone) to **Google Cloud Vertex AI Vector Search**, deployed serverlessly on **Vertex AI Reasoning Engine**.

-----

## 🏗️ 1. Infrastructure Provisioning (GCP Console)

Vertex AI utilizes a decoupled architecture for infinite scaling. You must provision the math (Index) separately from the compute (Endpoint).

1.  **Create the Vector Index:**
      * Navigate to **Vertex AI \> Vector Search \> Indexes**.
      * **Dimensions:** `768` (Required for `text-embedding-004`).
      * **Update Method:** `Stream` (Crucial for real-time data ingestion).
      * **Distance Measure:** `Dot product distance` (Faster/cheaper than Cosine for normalized embeddings).
2.  **Create & Deploy the Endpoint:**
      * Navigate to **Index Endpoints** and create a **Public Endpoint**.
      * Deploy your created Index to this Endpoint (Provisioning takes 15–30 mins).

-----

## 💻 2. Code Architecture & "Lazy" Initialization

Deploying code to Google Cloud Reasoning Engine requires specific patterns to avoid serialization crashes and LangChain validation errors.

### The Configuration (`.env`)

Use raw numeric IDs for your Vertex endpoints to prevent URL parsing errors.

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GCS_BUCKET_NAME=your-bucket-name

# Use exact numeric IDs from the GCP Console
VERTEX_INDEX_ID=824679900320366592
VERTEX_ENDPOINT_ID=1978604159531745280
```

### The Agent Implementation (`agent.py`)

To prevent the deployer from crashing while trying to "pickle" a live gRPC network connection, the database must be initialized **lazily** (only connecting when the first query is made).

```python
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from langchain_google_vertexai import VertexAIEmbeddings, VectorSearchVectorStore

load_dotenv()

# --- 1. LAZY DATABASE INITIALIZATION ---
_cached_vector_db = None

def get_vector_db():
    """Connects to Vertex AI only on the first request to survive cloud deployment."""
    global _cached_vector_db
    if _cached_vector_db is not None:
        return _cached_vector_db
        
    _cached_vector_db = VectorSearchVectorStore.from_components(
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
        region=os.getenv("GOOGLE_CLOUD_LOCATION"),
        index_id=os.getenv("VERTEX_INDEX_ID"),
        endpoint_id=os.getenv("VERTEX_ENDPOINT_ID"),
        embedding=VertexAIEmbeddings(model_name="text-embedding-004"),
        gcs_bucket_name=os.getenv("GCS_BUCKET_NAME") # Required by LangChain for v1 API
    )
    return _cached_vector_db

# --- 2. INTERNAL RAG TOOL ---
def search_internal_knowledge(query: str) -> str:
    try:
        db = get_vector_db()
        results = db.similarity_search(query, k=3)
        if not results:
            return "No relevant internal documents found."
            
        return "\n---\n".join([
            f"Source: {res.metadata.get('source', 'Unknown')}\nContent: {res.page_content}" 
            for res in results
        ])
    except Exception as e:
        return f"Warning: Database search failed: {str(e)}"

# --- 3. AGENT DEFINITION ---
root_agent = Agent(
    name="EnterpriseFactChecker",
    model="gemini-2.5-flash",
    instruction="You are an internal fact-checker...",
    tools=[search_internal_knowledge] 
)
```

-----

## 🚀 3. Deployment Protocol

1.  **Update `requirements.txt`:** Ensure all Google Cloud packages are present.
    ```text
    google-adk
    python-dotenv
    langchain-google-vertexai
    langchain-google-community[gcs]
    ```
2.  **Deploy Command:** Run from the same directory as your `agent.py`.
    ```bash
    adk deploy agent_engine .
    ```

-----

## 🚨 4. IAM Troubleshooting & Security Bypasses

Deploying to Reasoning Engine introduces strict Identity and Access Management (IAM) controls. If your agent returns a `403 Permission Denied` error during a database search, follow these protocols.

### Issue 1: LangChain Metadata Greediness

  * **Symptom:** `403 Permission 'aiplatform.indexes.get' denied`.
  * **Cause:** LangChain attempts to read the deep metadata of the Vector Index before executing a search. The standard `Vertex AI User` role blocks this action.
  * **Fix:** Upgrade the service account role from `Vertex AI User` to **`Vertex AI Administrator`**.

### Issue 2: The Hidden Reasoning Engine Service Account

  * **Symptom:** IAM permissions are set correctly on the default Compute Engine account, but 403 errors persist.
  * **Cause:** Reasoning Engine does not use your default service account. It provisions an isolated, hidden Google-managed service account specifically for your agent.
  * **Fix:** 1. Go to IAM and check **"Include Google-provided role grants"**.
    2\. Search for: `service-[YOUR_PROJECT_NUMBER]@gcp-sa-aiplatform-re.iam.gserviceaccount.com`.
    3\. Grant this specific account **Vertex AI Administrator** and **Storage Object Admin**.

### MLOps Pro-Tip: The Identity Interrogation Tool

If you are unsure which Service Account your cloud container is actively using, inject this tool into your agent to force it to read its own internal metadata server:

```python
def check_my_identity(query: str) -> str:
    """Forces the cloud server to reveal its exact IAM service account email."""
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
            headers={"Metadata-Flavor": "Google"}
        )
        with urllib.request.urlopen(req) as response:
            return f"My active Service Account is: {response.read().decode('utf-8')}"
    except Exception as e:
        return f"Error fetching identity: {str(e)}"
```

### Forcing a Cache Break

If you update IAM permissions but the container is caching an old security token, you must force a fresh build.

1.  Add a random comment to the bottom of `agent.py` (e.g., `# IAM Cache break 01`).
2.  Re-run `adk deploy agent_engine .`.
3.  The new container will wake up and grab a fresh, updated security token.