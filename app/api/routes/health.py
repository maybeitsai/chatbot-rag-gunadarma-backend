"""
Routes untuk health check dan monitoring
"""
from fastapi import APIRouter, Depends

from app.api.models import HealthResponse, StatsResponse
from app.api.dependencies import get_rag_pipeline
from app.api.services import HealthService, StatsService

router = APIRouter(tags=["health"])


@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check(rag_pipeline=Depends(get_rag_pipeline)):
    """Enhanced health check endpoint"""
    health_data = HealthService.check_rag_pipeline_health(rag_pipeline)
    
    return HealthResponse(
        status=health_data["status"],
        rag_pipeline=health_data["rag_pipeline"],
        database=health_data["database"],
        cache_enabled=health_data["cache_enabled"],
        optimized=health_data["optimized"],
        stats=health_data["stats"]
    )


@router.get("/api/v1/stats")
async def get_stats(rag_pipeline=Depends(get_rag_pipeline)):
    """Get comprehensive system statistics"""
    return StatsService.get_system_stats(rag_pipeline)