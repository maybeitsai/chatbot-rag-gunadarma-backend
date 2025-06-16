import os
import json
import pandas as pd
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

class DataProcessor:
    def __init__(self):
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 500))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load data from JSON or CSV file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV.")
        
        return data
    
    def create_documents(self, data: List[Dict]) -> List[Document]:
        """Convert raw data to LangChain documents"""
        documents = []
        
        for item in data:
            # Skip items without text content
            if not item.get('text_content') or item['text_content'].strip() == '':
                continue
            
            # Create document with metadata
            doc = Document(
                page_content=item['text_content'],
                metadata={
                    'url': item.get('url', ''),
                    'title': item.get('title', ''),
                    'source_type': item.get('source_type', ''),
                    'timestamp': item.get('timestamp', '')
                }
            )
            documents.append(doc)
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunked_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                )
                chunked_docs.append(chunked_doc)
        
        return chunked_docs
    
    def process_data(self, file_path: str) -> List[Document]:
        """Complete data processing pipeline"""
        print(f"Loading data from {file_path}...")
        raw_data = self.load_data(file_path)
        print(f"Loaded {len(raw_data)} items")
        
        print("Creating documents...")
        documents = self.create_documents(raw_data)
        print(f"Created {len(documents)} documents")
        
        print("Chunking documents...")
        chunked_documents = self.chunk_documents(documents)
        print(f"Created {len(chunked_documents)} chunks")
        
        return chunked_documents

if __name__ == "__main__":
    processor = DataProcessor()
    
    # Try to process JSON first, then CSV
    for file_path in ["data/output.json", "data/output.csv"]:
        if os.path.exists(file_path):
            documents = processor.process_data(file_path)
            print(f"Successfully processed {len(documents)} document chunks")
            break
    else:
        print("No data files found. Please ensure data/output.json or data/output.csv exists.")