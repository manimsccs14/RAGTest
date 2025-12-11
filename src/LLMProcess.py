from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

# Load Groq API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM
llm = ChatGroq(api_key=groq_api_key, temperature=0.1, max_tokens=1024, model="llama-3.1-8b-instant")

## Simple RAG function: retrieve the context + generate the response
def rag_qa(query, retriever, llm) -> str:
    """ 
    Perform RAG-based question answering.
    
    Args:
         query (str): The query string to answer.
         retriever (RAGRetriever): Instance of RAGRetriever to retrieve relevant documents.
         llm (ChatGroq): Instance of ChatGroq for generating responses.
    Returns:
             str: Generated answer from the LLM.
    """
    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(query)
    if not retrieved_docs:
        return "No relevant documents found."
    
    print(retrieved_docs)
    # Combine retrieved documents into context
    context = "\n".join([doc['document'] for doc in retrieved_docs])

    if not context.strip():
        return "No relevant content found in the retrieved documents."
    
    # Create prompt for LLM
    prompt = """ Use the following context to answer the question concisely.
                Context: {context}

                Question: {query}

                Answer:
        """
    
    # Generate response using LLM
    response = llm.invoke([prompt.format(context=context, query=query)])
    return response.content

# Example usage
if __name__ == "__main__":
    from VectorStore import VectorStore
    from EmbeddingManager import EmbeddingManager
    from RAGRetriever import RAGRetriever
    vector_store = VectorStore()
    embedding_manager = EmbeddingManager()
    rag_retriever = RAGRetriever(vector_store, embedding_manager)
    answer = rag_qa("What is angular", rag_retriever, llm)
    print(answer)