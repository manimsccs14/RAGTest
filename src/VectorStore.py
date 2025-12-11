import numpy as np
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
import FileLoader
import EmbeddingManager
import os

class VectorStore:
    """ Vector Store class to handle storage and retrieval of document embeddings. """
    def __init__(self,collection_name: str = "pdf", persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = None  # Placeholder for actual Chroma client initialization
        self.collection_name = collection_name
        self._initialize_store()

    def _initialize_store(self):
        """ Initialize the Chroma client and collection. """
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path= self.persist_directory)

        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
        print(f"ChromaDB initialized at '{self.persist_directory}' with collection '{self.collection_name}'.")  

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """ 
        Add documents to the vector store after generating embeddings.
        
        Args:
             documents (List[Document]): List of Document objects to add.
             embedding_manager (EmbeddingManager): Instance of EmbeddingManager to generate embeddings.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [str(uuid.uuid4()) for _ in documents]

        # embeddings = embedding_manager.generate_embedding(texts)

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts
        )
        print(f"Added {len(documents)} documents to the vector store.")
        print(f"Vector Store initialized with collection '{self.collection_name}'.")

# Example usage
if __name__ == "__main__":
    vector_store = VectorStore()
    fileloader = FileLoader.FileLoader()
    documents = fileloader.load_directory()
    print(f"Loaded {len(documents)} documents from directory.")
    embedding_manager = EmbeddingManager.EmbeddingManager()
    chunks = embedding_manager.chunk_documents(documents)
    #print(chunks)
    text = [chunk.page_content for chunk in chunks]
    embedding = embedding_manager.generate_embedding(text)
    print(f"Generated embeddings with shape: {embedding.shape}")
    vector_store.add_documents(chunks, embedding)
    print("Vector Store is ready for use.")