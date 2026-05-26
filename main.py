import os
from openai import AzureOpenAI
import chromadb

# Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint="https://aoai-rag-demo.openai.azure.com/"
)

# Chroma DB setup
chroma_client = chromadb.PersistentClient(path="./db")
collection = chroma_client.create_collection("docs")

# Example document
text = "Azure Cognitive Services provides AI models for enterprise use."

# Generate embedding
response = client.embeddings.create(
    model="embedding-deploy",  # your embedding deployment name
    input=text
)
vector = response.data[0].embedding

# Store in Chroma
collection.add(documents=[text], embeddings=[vector], ids=["doc1"])
print("✅ Embedding stored in Chroma DB")
