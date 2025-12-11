import numpy as np
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import FileLoader


class EmbeddingManager:
    """ Embedding Manager class to handle document embeddings. """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = None  # Placeholder for actual model initialization
        self._load_model()

    def _load_model(self):
        """ Load the embedding model. """
        self.model = SentenceTransformer(self.model_name)
        print(f"Model '{self.model_name}' loaded successfully.")
        print(f"Embedding dimentions: {self.model.get_sentence_embedding_dimension()} dimensions.")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    
    def generate_embedding(self, text: List[Any]) -> np.ndarray:
        """ 
        Generate embeddings for the given text.
        
        Args:
             text (List[str]): List of text strings to generate embeddings for.
        Returns:
                 np.ndarray: Array of embeddings.
                 
        """
        # text = [chunk.page_content for chunk in chunks]
        print(f"Generated embeddings for {len(text)} texts.")

        embedding = self.model.encode(text, show_progress_bar=True)
        print(f"Embedding shape: {embedding.shape}")
        return embedding

# Example usage
if __name__ == "__main__":
    fileloader = FileLoader.FileLoader()
    documents = fileloader.load_directory()
    print(f"Loaded {len(documents)} documents from directory.")
    embedding_manager = EmbeddingManager()
    chunks = embedding_manager.chunk_documents(documents)
    print(chunks)
    text = [chunk.page_content for chunk in chunks]
    embedding = embedding_manager.generate_embedding(text)
    print(f"Generated embeddings with shape: {embedding.shape}")
# Sample TXT Document Creation