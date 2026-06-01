import os
from openai import AzureOpenAI
import chromadb

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint="https://aoai-rag-demo.openai.azure.com/"
)

chroma_client = chromadb.PersistentClient(path="./db")
collection = chroma_client.get_collection("docs")

query = "What does Azure Cognitive Services provide?"

query_embedding = client.embeddings.create(
    model="embedding-deploy",
    input=query
).data[0].embedding

results = collection.query(query_embeddings=[query_embedding], n_results=3)
context = " ".join([doc for doc in results['documents'][0]])

response = client.chat.completions.create(
    model="gpt-35-deploy",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Answer using context: {context}\n\nQuestion: {query}"}
    ]
)

print("💡 Answer:", response.choices[0].message["content"])
