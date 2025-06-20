"""
Routes untuk cache management
"""
from fastapi import APIRouter, Depends

from app.api.models import CacheResponse
from app.api.dependencies import get_rag_pipeline
from app.api.services import CacheService

router = APIRouter(prefix="/api/v1/cache", tags=["cache"])


@router.post("/api/v1/clear", response_model=CacheResponse)
async def clear_cache(rag_pipeline=Depends(get_rag_pipeline)):
    """Clear semantic cache"""
    result = CacheService.clear_cache(rag_pipeline)
    return CacheResponse(message=result["message"])


@router.get("/api/v1/stats")
async def get_cache_stats(rag_pipeline=Depends(get_rag_pipeline)):
    """Get cache statistics"""
    return CacheService.get_cache_stats(rag_pipeline)