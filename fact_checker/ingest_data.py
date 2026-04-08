import os
from dotenv import load_dotenv
from langchain_google_community import GCSDirectoryLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# 1. Load the secrets from your .env file
load_dotenv()

# 2. Fetch the secrets securely
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# UPDATED: Matching your exact Pinecone index name
INDEX_NAME = "gcp-vertex-ai" 

print("🔍 Reading documents from Cloud Storage...")
loader = GCSDirectoryLoader(project_name=PROJECT_ID, bucket=BUCKET_NAME)
docs = loader.load()

print("✂️ Chunking documents...")
# We split the document into 1000-character chunks. 
# The 100-character overlap ensures we don't accidentally cut a sentence in half!
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

print("🧠 Generating embeddings and uploading to Pinecone...")
# This Google model outputs exactly 768 dimensions!
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

vector_db = PineconeVectorStore.from_documents(
    chunks, 
    embeddings, 
    index_name=INDEX_NAME, 
    pinecone_api_key=PINECONE_API_KEY
)

print("✅ Data successfully ingested into Pinecone!")