"""
FastAPI dependencies untuk validasi dan dependency injection
"""
from fastapi import Depends
from app.api.exceptions import (
    RAGPipelineNotInitialized,
    ApplicationStartupIncomplete
)


class AppState:
    """Singleton untuk menyimpan state aplikasi"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.rag_pipeline = None
            cls._instance.app_startup_complete = False
        return cls._instance
    
    def set_rag_pipeline(self, pipeline):
        self.rag_pipeline = pipeline
    
    def set_startup_complete(self, status: bool):
        self.app_startup_complete = status
    
    def get_rag_pipeline(self):
        return self.rag_pipeline
    
    def is_startup_complete(self) -> bool:
        return self.app_startup_complete


def get_app_state() -> AppState:
    """Dependency untuk mendapatkan app state"""
    return AppState()


def get_rag_pipeline(app_state: AppState = Depends(get_app_state)):
    """Dependency untuk mendapatkan RAG pipeline dengan validasi"""
    if not app_state.is_startup_complete():
        raise ApplicationStartupIncomplete()
    
    if app_state.get_rag_pipeline() is None:
        raise RAGPipelineNotInitialized()
    
    return app_state.get_rag_pipeline()


def validate_startup_complete(app_state: AppState = Depends(get_app_state)):
    """Dependency untuk memvalidasi startup complete"""
    if not app_state.is_startup_complete():
        raise ApplicationStartupIncomplete()
    return True