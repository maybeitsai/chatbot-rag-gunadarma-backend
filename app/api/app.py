"""
FastAPI application factory dan configuration
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import AppState
from app.api.routes import question, health, cache, root, websocket
from app.api.config import settings
from app.api.middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler untuk startup dan shutdown"""
    # Startup
    app_state = AppState()
    
    logger.info("Starting RAG Pipeline initialization...")
    
    try:
        # Import pipeline
        from app.rag.pipeline import create_rag_pipeline
        
        # Initialize RAG pipeline
        rag_pipeline = create_rag_pipeline(enable_cache=settings.ENABLE_CACHE)
        app_state.set_rag_pipeline(rag_pipeline)
        logger.info("Using optimized RAG Pipeline with caching")
        
        # Test connection
        try:
            if hasattr(rag_pipeline, 'test_connection_async'):
                is_connected = await rag_pipeline.test_connection_async()
            else:
                # For pipelines without async test_connection method
                is_connected = True
                
            if is_connected:
                logger.info("RAG Pipeline initialized successfully!")
                
                # Warm up cache with common questions if using optimized pipeline
                if hasattr(rag_pipeline, 'warmup_cache'):
                    warmup_questions = settings.get_warmup_questions()
                    await rag_pipeline.warmup_cache(warmup_questions)
                    logger.info("Cache warmup completed")
                
                app_state.set_startup_complete(True)
            else:
                logger.error("RAG Pipeline connection test failed")
                
        except Exception as conn_error:
            logger.error(f"Connection test error: {conn_error}")
            # Continue anyway, mark as ready for basic operation
            app_state.set_startup_complete(True)
            
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {e}")
        app_state.set_rag_pipeline(None)
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Pipeline...")
    rag_pipeline = app_state.get_rag_pipeline()
    if rag_pipeline and hasattr(rag_pipeline, 'cleanup'):
        rag_pipeline.cleanup()
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Factory function untuk membuat FastAPI app"""
    app = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        version=settings.API_VERSION,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)    # Include routers
    app.include_router(root.router)
    app.include_router(question.router)
    app.include_router(health.router)
    app.include_router(cache.router)
    app.include_router(websocket.router)
    
    return app