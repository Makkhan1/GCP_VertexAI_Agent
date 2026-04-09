# Using Pinecone vector database to ingest data from GCP vertex ai.

# import os
# from dotenv import load_dotenv
# from langchain_google_community import GCSDirectoryLoader
# from langchain_google_vertexai import VertexAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_pinecone import PineconeVectorStore

# # 1. Load the secrets from your .env file
# load_dotenv()

# # 2. Fetch the secrets securely
# PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
# BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# # UPDATED: Matching your exact Pinecone index name
# INDEX_NAME = "gcp-vertex-ai" 

# print("🔍 Reading documents from Cloud Storage...")
# loader = GCSDirectoryLoader(project_name=PROJECT_ID, bucket=BUCKET_NAME)
# docs = loader.load()

# print("✂️ Chunking documents...")
# # We split the document into 1000-character chunks. 
# # The 100-character overlap ensures we don't accidentally cut a sentence in half!
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
# chunks = text_splitter.split_documents(docs)

# print("🧠 Generating embeddings and uploading to Pinecone...")
# # This Google model outputs exactly 768 dimensions!
# embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

# vector_db = PineconeVectorStore.from_documents(
#     chunks, 
#     embeddings, 
#     index_name=INDEX_NAME, 
#     pinecone_api_key=PINECONE_API_KEY
# )

# print("✅ Data successfully ingested into Pinecone!")

## Using GCP vertex ai to ingest data into pinecone vector database.  

import os
from dotenv import load_dotenv
from langchain_google_community import GCSDirectoryLoader
from langchain_google_vertexai import VertexAIEmbeddings, VectorSearchVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
INDEX_ID = os.getenv("VERTEX_INDEX_ID")
ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID")

print("🔍 Reading docs from GCS...")
loader = GCSDirectoryLoader(project_name=PROJECT_ID, bucket=BUCKET_NAME)
docs = loader.load()

print("✂️ Chunking...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

print("🧠 Initializing Vertex AI Vector Search Connection...")
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

# STEP 1: Establish the connection using the specific GCP method
vector_db = VectorSearchVectorStore.from_components(
    project_id=PROJECT_ID,
    region=LOCATION,
    index_id=INDEX_ID,
    endpoint_id=ENDPOINT_ID,
    embedding=embeddings,
    stream_update=True,  # Required because you built a Streaming Index!
    gcs_bucket_name=BUCKET_NAME
)

print("📤 Uploading chunks to the Endpoint...")
# STEP 2: Add the documents explicitly
vector_db.add_documents(chunks)

print("✅ Data successfully ingested into Vertex AI!")