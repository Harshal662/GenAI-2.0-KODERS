# query_bot.py
from pinecone_setup import initialize_pinecone
from llama_inference import initialize_llama_client, generate_response
import requests

# Azure OpenAI API setup
AZURE_API_KEY = "a6a181d997c648269387f0c3a6ccaba0"
AZURE_API_VERSION = "2023-05-15"
AZURE_ENDPOINT = "https://genai-openai-koders.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings"

def get_azure_embedding(text):
    """Generate embeddings using Azure OpenAI API."""
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY,
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-large",
    }
    
    response = requests.post(f"{AZURE_ENDPOINT}?api-version={AZURE_API_VERSION}", headers=headers, json=payload)
    
    if response.status_code == 200:
        embedding = response.json()["data"][0]["embedding"]
        return embedding
    else:
        print(f"Error generating embedding: {response.status_code} - {response.text}")
        return None

def query_bot_response(query,chat_history):
    # Initialize Pinecone and ChatGroq
    index = initialize_pinecone()
    client = initialize_llama_client()

    # Generate embedding for the user query
    query_embedding = get_azure_embedding(query)
    if query_embedding is None:
        return "Error: Unable to process query embedding."

    # Query Pinecone for relevant documents based on the query embedding
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    context = " ".join([res['metadata'].get('completion', '') for res in results['matches']])

    # Generate response using ChatGroq with the retrieved context and original query
    response = generate_response(client, query, context,chat_history)
    print(response)
    return response
