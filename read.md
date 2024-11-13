For our hackathon, we were tasked with enhancing customer experiences using personalized financial assistance. Our approach involved creating a RAG-based financial advice chatbot (RAG stands for Retrieval-Augmented Generation). The chatbot was designed to provide personalized financial advice using user data and relevant documents.

We used Pinecone as the vector database to store both user data and documents from Citibank related to different financial schemes. This helped us deliver personalized results without exposing user data to LLM (Large Language Models), ensuring data privacy and security.

For the LLM, we integrated Llama 2 model using Groq AI for inferencing, which allowed us to access and run LLMs locally with low latency. This was particularly beneficial since it reduced dependency on cloud infrastructure and minimized delays. While we used open-source models during the hackathon, we considered deploying the solution to the cloud for broader scalability.

In terms of standing out, a key strategy for us was focusing on a practical, real-world use case—making financial advice more accessible and secure. We also paid attention to privacy concerns by ensuring sensitive data wasn't directly exposed to the models. 

Feel free to reach out if you need more details about the code or architecture! 