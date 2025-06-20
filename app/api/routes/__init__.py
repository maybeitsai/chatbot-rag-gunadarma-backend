"""
API routes module

Module ini berisi semua route handlers untuk API endpoints.
"""

from . import question, health, cache, root

# Import routers untuk mudah diakses
from .question import router as question_router
from .health import router as health_router
from .cache import router as cache_router
from .root import router as root_router

__all__ = [
    # Modules
    'question',
    'health',
    'cache',
    'root',
    
    # Routers
    'question_router',
    'health_router',
    'cache_router',
    'root_router'
]
