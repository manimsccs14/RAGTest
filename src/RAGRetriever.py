import numpy as np
import uuid
import EmbeddingManager
import VectorStore
from typing import List, Dict, Any


class RAGRetriever:
    """ RAG Retriever class to handle retrieval of relevant documents based on queries. """
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager, top_k: int = 5):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """ 
        Retrieve relevant documents based on the query.
        
        Args:
             query (str): The query string to search for.
        Returns:
                 List[Dict[str, Any]]: List of retrieved documents with metadata.
        """
        query_embedding = self.embedding_manager.generate_embedding([query])
        
        results = self.vector_store.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=self.top_k
        )
        
        retrieved_docs = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            retrieved_docs.append({
                "document": doc,
                "metadata": metadata
            })
        
        print(f"Retrieved {len(retrieved_docs)} documents for the query: '{query}'")
        return retrieved_docs
    
# Example usage
if __name__ == "__main__":
    vector_store = VectorStore.VectorStore()
    embedding_manager = EmbeddingManager.EmbeddingManager()
    rag_retriever = RAGRetriever(vector_store, embedding_manager)
    
    sample_query = "What is the capital of France?"
    retrieved_documents = rag_retriever.retrieve(sample_query)
    for idx, doc in enumerate(retrieved_documents):
        print(f"Document {idx+1}: {doc['document']}\nMetadata: {doc['metadata']}\n")