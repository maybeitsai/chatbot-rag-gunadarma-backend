"""
Main routes untuk aplikasi
"""
from fastapi import APIRouter, Depends

from app.api.dependencies import get_app_state, AppState
from app.api.models import ExampleQuestionsResponse

router = APIRouter()


@router.get("/")
async def root(app_state: AppState = Depends(get_app_state)):
    """Root endpoint with startup status"""
    return {
        "message": "Gunadarma RAG API - Optimized",
        "status": "running" if app_state.is_startup_complete() else "starting",
        "version": "2.0.0",
        "features": [
            "Semantic Caching",
            "Optimized Vector Store with HNSW",
            "Asynchronous Processing",
            "Metadata Filtering",
            "Performance Monitoring"
        ]
    }


@router.get("/api/v1/examples", response_model=ExampleQuestionsResponse)
async def get_example_questions():
    """Get example questions that can be asked"""
    return ExampleQuestionsResponse(
        example_questions=[
            "Apa itu Universitas Gunadarma?",
            "Fakultas apa saja yang ada di Universitas Gunadarma?",
            "Bagaimana cara mendaftar di Universitas Gunadarma?",
            "Apa saja program studi yang tersedia?",
            "Dimana lokasi kampus Universitas Gunadarma?",
            "Bagaimana sistem pembelajaran di Universitas Gunadarma?",
            "Apa saja fasilitas yang tersedia di kampus?",
            "Bagaimana cara menghubungi BAAK Universitas Gunadarma?"
        ]
    )