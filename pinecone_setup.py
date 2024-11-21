# pinecone_setup.py
import os
from pinecone import Pinecone, ServerlessSpec

def initialize_pinecone():
    # Create an instance of Pinecone
    pc = Pinecone(api_key='5921d86c-5564-4752-8f15-195e37773e89')
    
    # Define index name and dimension
    index_name = "testing1"
    dimension = 3072  # Adjust based on model output dimensions if different 1536

    # Check and create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    # Retrieve and return the index
    index = pc.Index(index_name)
    return index
