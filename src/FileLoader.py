from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from typing import List

class FileLoader:
    """
    File Loader class to load documents from a specified directory.
    """
    def __init__(self, dir_path: str = "./data/pdf", 
                 dir_glob: str = "*.pdf",
                loader_cls=PyPDFLoader):
        self.dir_path = dir_path
        self.dir_glob = dir_glob
        self.loader_cls = loader_cls
        # self._load_directory()

    def load_directory(self) -> List[Document]:
        loader = DirectoryLoader(self.dir_path, glob=self.dir_glob,
                                loader_cls=self.loader_cls)
        documents = loader.load()
        return documents

# Example usage
if __name__ == "__main__":
    file_loader = FileLoader()
    documents = file_loader.load_directory()
    print(f"Loaded {len(documents)} documents from {file_loader.dir_path} matching {file_loader.dir_glob}.")