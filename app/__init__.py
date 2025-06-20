"""
Main application package untuk RAG chatbot system

Package ini berisi semua modul utama:
- api: FastAPI application dan routes
- rag: RAG pipeline dan processing
- crawl: Web crawling functionality
"""

from . import api, rag, crawl

__version__ = "1.0.0"

__all__ = [
    'api',
    'rag', 
    'crawl',
    '__version__'
]
