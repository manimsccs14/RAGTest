
import os
from dotenv import load_dotenv
load_dotenv()

print("Environment variables loaded.")

print(f"USER_AGENT: {os.getenv('USER_AGENT')}")

class URLLoader:
    def __init__(self):
        pass

    def load_url(self, url):
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(url)
        documents = loader.load()
        return documents
    
# Example usage
if __name__ == "__main__":
    url_loader = URLLoader()
    url = "https://www.citigroup.com/global"
    documents = url_loader.load_url(url)
    print(documents)
    print(f"Loaded {len(documents)} documents from URL: {url}.")