# llama_inference.py
from langchain_groq import ChatGroq

def initialize_llama_client():
    try:
        # Initialize ChatGroq client with API key and model
        client = ChatGroq(
            model="mixtral-8x7b-32768",
            api_key='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',
            temperature=0.8,
            max_tokens=1024
        )
        print("ChatGroq client initialized successfully.")
        return client
    except Exception as e:
        print(f"Error initializing ChatGroq client: {e}")
        return None

def generate_response(client, query, context="",chat_history=[]):
    if client is None:
        return "Error: Client initialization failed. Cannot generate response."

    try:
        # Prepare messages for completion
        messages = [
            ("system", context),
            ("human", query)
        ]
        
        
        # Generate chat completion
        ai_msg = client.invoke(messages)
        return ai_msg.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error: Unable to generate response."
