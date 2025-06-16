#!/usr/bin/env python3
"""
Setup script untuk inisialisasi pipeline RAG
"""

import os
import sys
import asyncio
from crawl.crawler import crawl_pipeline
from rag.vector_store import VectorStoreManager

def main():
    """Main setup function"""
    print("=== Gunadarma RAG Pipeline Setup ===")
    
    # Crawl data from Gunadarma websites
    print("Starting data crawling...")
    asyncio.run(crawl_pipeline())

    # Check if data files exist
    data_files = ["data/output.json", "data/output.csv"]
    data_file = None
    
    for file_path in data_files:
        if os.path.exists(file_path):
            data_file = file_path
            break
    
    if not data_file:
        print("ERROR: No data files found!")
        print("Please ensure one of these files exists:")
        for file_path in data_files:
            print(f"  - {file_path}")
        sys.exit(1)
    
    print(f"Found data file: {data_file}")
    
    # Check environment variables
    required_env = [
        "GOOGLE_API_KEY",
        "NEON_CONNECTION_STRING"
    ]
    
    missing_env = []
    for env_var in required_env:
        if not os.getenv(env_var):
            missing_env.append(env_var)
    
    if missing_env:
        print("ERROR: Missing required environment variables:")
        for var in missing_env:
            print(f"  - {var}")
        print("Please check your .env file")
        sys.exit(1)
    
    print("Environment variables OK")
    
    # Initialize vector store
    try:
        print("Initializing vector store manager...")
        manager = VectorStoreManager()
        
        print("Setting up database and populating vector store...")
        vector_store = manager.setup_and_populate(data_file)
        
        if vector_store:
            print("✅ Setup completed successfully!")
            print("You can now run the API with: python main.py")
        else:
            print("❌ Setup failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()