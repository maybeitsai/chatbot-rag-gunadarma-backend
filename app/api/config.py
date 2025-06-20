"""
Configuration management untuk API
"""
import os
from typing import Optional


class Settings:
    """Application settings dari environment variables"""
    
    # Server configuration
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    RELOAD: bool = os.getenv("RELOAD", "true").lower() == "true"
    
    # RAG Pipeline configuration
    EMBEDDING_MODEL: Optional[str] = os.getenv("EMBEDDING_MODEL")
    LLM_MODEL: Optional[str] = os.getenv("LLM_MODEL")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))
    
    # Cache configuration
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    
    # Batch processing limits
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", 10))
    
    # Logging configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # API configuration
    API_TITLE: str = "Gunadarma RAG API"
    API_DESCRIPTION: str = "Enhanced Retrieval-Augmented Generation API untuk informasi Universitas Gunadarma dengan semantic caching dan optimasi performa"
    API_VERSION: str = "2.0.0"
    
    # CORS configuration
    CORS_ORIGINS: list = ["*"]  # In production, specify allowed origins
    
    @classmethod
    def get_warmup_questions(cls) -> list[str]:
        """Get default warmup questions"""
        return [
            "Apa itu Universitas Gunadarma?",
            "Fakultas apa saja yang ada di Universitas Gunadarma?",
            "Bagaimana cara mendaftar di Universitas Gunadarma?",
            "Dimana lokasi kampus Universitas Gunadarma?"
        ]
    
    @classmethod
    def get_example_questions(cls) -> list[str]:
        """Get example questions for API documentation"""
        return [
            "Apa itu Universitas Gunadarma?",
            "Fakultas apa saja yang ada di Universitas Gunadarma?",
            "Bagaimana cara mendaftar di Universitas Gunadarma?",
            "Apa saja program studi yang tersedia?",
            "Dimana lokasi kampus Universitas Gunadarma?",
            "Bagaimana sistem pembelajaran di Universitas Gunadarma?",
            "Apa saja fasilitas yang tersedia di kampus?",
            "Bagaimana cara menghubungi BAAK Universitas Gunadarma?"
        ]


# Global settings instance
settings = Settings()