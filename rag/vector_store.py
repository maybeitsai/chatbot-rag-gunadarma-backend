import os
from typing import List
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv
from rag.db_setup import setup_database
from rag.data_processor import DataProcessor

load_dotenv()

class VectorStoreManager:
    def __init__(self):
        self.connection_string = os.getenv("NEON_CONNECTION_STRING")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model,
            google_api_key=self.google_api_key
        )
        
        # Collection name for PGVector
        self.collection_name = "chatbot-gunadarma"
        
    def initialize_vector_store(self) -> PGVector:
        """Initialize PGVector store"""
        return PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection_string,
            use_jsonb=True,
        )
    
    def setup_and_populate(self, data_file_path: str):
        """Setup database and populate with documents"""
        print("Setting up database...")
        setup_database()
        
        print("Processing data...")
        processor = DataProcessor()
        documents = processor.process_data(data_file_path)
        
        if not documents:
            print("No documents to process!")
            return None
        
        print("Initializing vector store...")
        vector_store = self.initialize_vector_store()
        
        print(f"Adding {len(documents)} documents to vector store...")
        # Add documents in batches to avoid memory issues
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)
            print(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        print("Vector store populated successfully!")
        return vector_store
    
    def get_vector_store(self) -> PGVector:
        """Get existing vector store"""
        return self.initialize_vector_store()

if __name__ == "__main__":
    manager = VectorStoreManager()
    
    # Find data file
    data_file = None
    for file_path in ["data/output.json", "data/output.csv"]:
        if os.path.exists(file_path):
            data_file = file_path
            break
    
    if data_file:
        manager.setup_and_populate(data_file)
    else:
        print("No data files found. Please ensure data/output.json or data/output.csv exists.")