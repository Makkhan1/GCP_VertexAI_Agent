# 🧠 Vertex AI Reasoning Engine: Dual-Engine RAG Fact-Checker

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Vertex_AI-4285F4.svg)
![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-lightgrey.svg)
![Tavily](https://img.shields.io/badge/Search-Tavily_AI-green.svg)

An enterprise-grade, serverless AI agent deployed on **Google Cloud Vertex AI**. This project demonstrates a **Hybrid Retrieval-Augmented Generation (RAG)** architecture. 

Due to Vertex AI's strict API limitations regarding mixing Native Grounding tools with Custom Python Functions, this agent utilizes an "All-Custom" tool architecture. It autonomously routes queries to either a private Pinecone Vector Database for internal company knowledge or the Tavily API for real-time global web search.

---

## 🏗️ Architecture

* **Framework:** Google Cloud Agent Development Kit (ADK)
* **Compute:** Vertex AI Reasoning Engines (Serverless containerization)
* **LLM:** `gemini-2.5-flash`
* **Internal Memory (RAG):** Pinecone (768-Dimension Dense Index, Cosine Metric)
* **External Memory (Web):** Tavily AI Search API
* **Embeddings:** `text-embedding-004` (via LangChain)

---

## 🚀 Prerequisites

1. A **Google Cloud Project** with Vertex AI enabled.
2. A **Google Cloud Storage (GCS)** Bucket for raw document storage.
3. A **Pinecone** API Key and an empty 768-dimension index.
4. A **Tavily** API Key.

---

## 🛠️ Environment Setup

**1. Install Dependencies**
Ensure your `requirements.txt` includes the following:
```text
google-adk
python-dotenv
langchain-pinecone
langchain-google-vertexai
langchain-google-community[gcs]
tavily-python
```
Run `pip install -r requirements.txt`.

**2. Secure Credentials (`.env`)**
Create a `.env` file in your project root. **Never commit this to version control.**
```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GCS_BUCKET_NAME=your-bucket-name

# APIs
PINECONE_API_KEY=your-pinecone-key
TAVILY_API_KEY=tvly-your-tavily-key

# Observability
GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY=true
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

---

## 🗄️ Phase 1: Data Ingestion (The Data Lake to Pinecone Pipeline)

To populate the private knowledge base, upload your text documents or PDFs to your GCS Bucket, then run the ingestion script. This script chunks the data, converts it into 768-dimension math using Google's embedding model, and loads it into Pinecone.

**File: `ingest_data.py`**
```python
import os
from dotenv import load_dotenv
from langchain_google_community import GCSDirectoryLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "gcp-vertex-ai" # Must be exactly 768 dimensions

print("🔍 Reading documents from Cloud Storage...")
loader = GCSDirectoryLoader(project_name=PROJECT_ID, bucket=BUCKET_NAME)
docs = loader.load()

print("✂️ Chunking documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

print("🧠 Generating embeddings and uploading to Pinecone...")
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
vector_db = PineconeVectorStore.from_documents(
    chunks, embeddings, index_name=INDEX_NAME, pinecone_api_key=PINECONE_API_KEY
)

print("✅ Data successfully ingested into Pinecone!")
```

Run the pipeline: `python ingest_data.py`

---

## 🤖 Phase 2: The Hybrid Agent

This agent features conditional routing. It is instructed to handle conversational greetings naturally, use Pinecone for internal queries, and use Tavily for public web searches.

**File: `agent.py`**
```python
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from langchain_pinecone import PineconeVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from tavily import TavilyClient

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

# --- TOOL 2: EXTERNAL WEB SEARCH ---
def web_search(query: str) -> str:
    """Searches the public internet for up-to-date news and global facts."""
    api_key = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, search_depth="basic")
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
    2. For company-specific or internal questions, use 'search_internal_knowledge'.
    3. For world news or public facts, use 'web_search'.
    
    VERDICT FORMAT:
    VERDICT: [True / False / Unverified]
    REASONING: [Provide explanation]
    """,
    tools=[search_internal_knowledge, web_search] 
)
```

---

## ☁️ Deployment & Testing

**1. Local Testing**
Test the agent's routing logic locally before deploying:
```bash
adk web
```
*Access the local UI at `http://127.0.0.1:8000`.*

> **Troubleshooting Tip:** If `adk web` hangs on startup or throws a "stale session" error, delete the hidden `.adk` folder in your project directory and restart the server. This resets the local SQLite development database.

**2. Cloud Deployment**
Deploy the agent to a serverless container on Vertex AI:
```bash
adk deploy agent_engine .
```
```

***

Now that your RAG pipeline and documentation are pristine, are you ready to tackle Level 2 (Permanent Memory/BigQuery) or Level 3 (Building a Streamlit UI)?