from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import asyncio
import logging
import time
from contextlib import asynccontextmanager

# Import pipeline
from rag.pipeline import create_rag_pipeline
from rag.semantic_cache import get_semantic_cache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Global variables
rag_pipeline = None
app_startup_complete = False

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_pipeline, app_startup_complete
    
    logging.info("Starting RAG Pipeline initialization...")
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = create_rag_pipeline(enable_cache=True)
        logging.info("Using optimized RAG Pipeline with caching")
        
        # Test connection
        try:
            if hasattr(rag_pipeline, 'test_connection_async'):
                is_connected = await rag_pipeline.test_connection_async()
            else:
                # For pipelines without async test_connection method
                is_connected = True
                
            if is_connected:
                logging.info("RAG Pipeline initialized successfully!")
                
                # Warm up cache with common questions if using optimized pipeline
                if hasattr(rag_pipeline, 'warmup_cache'):
                    warmup_questions = [
                        "Apa itu Universitas Gunadarma?",
                        "Fakultas apa saja yang ada di Universitas Gunadarma?",
                        "Bagaimana cara mendaftar di Universitas Gunadarma?",
                        "Dimana lokasi kampus Universitas Gunadarma?"
                    ]
                    await rag_pipeline.warmup_cache(warmup_questions)
                    logging.info("Cache warmup completed")
                
                app_startup_complete = True
            else:
                logging.error("RAG Pipeline connection test failed")
                
        except Exception as conn_error:
            logging.error(f"Connection test error: {conn_error}")
            # Continue anyway, mark as ready for basic operation
            app_startup_complete = True
            
    except Exception as e:
        logging.error(f"Failed to initialize RAG Pipeline: {e}")
        rag_pipeline = None
    
    yield
    
    # Shutdown
    logging.info("Shutting down RAG Pipeline...")
    if rag_pipeline and hasattr(rag_pipeline, 'cleanup'):
        rag_pipeline.cleanup()
    logging.info("Application shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Gunadarma RAG API",
    description="Enhanced Retrieval-Augmented Generation API untuk informasi Universitas Gunadarma dengan semantic caching dan optimasi performa",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    use_cache: Optional[bool] = True
    metadata_filter: Optional[Dict[str, Any]] = None

class QuestionResponse(BaseModel):
    answer: str
    source_urls: List[str]
    status: str
    source_count: Optional[int] = 0
    response_time: Optional[float] = None
    cached: Optional[bool] = False
    cache_type: Optional[str] = None

class BatchQuestionRequest(BaseModel):
    questions: List[str]
    use_cache: Optional[bool] = True

class BatchQuestionResponse(BaseModel):
    results: List[QuestionResponse]
    total_questions: int
    processing_time: float

# Note: RAG pipeline initialization happens in the lifespan event handler above

@app.get("/")
async def root():
    """Root endpoint with startup status"""
    return {
        "message": "Gunadarma RAG API - Optimized",
        "status": "running" if app_startup_complete else "starting",
        "version": "2.0.0",
        "features": [
            "Semantic Caching",
            "Optimized Vector Store with HNSW",
            "Asynchronous Processing",
            "Metadata Filtering",
            "Performance Monitoring"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    if not app_startup_complete:
        raise HTTPException(status_code=503, detail="Application is still starting up")
    
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    # Test connection
    try:
        is_healthy = rag_pipeline.test_connection()
        
        if not is_healthy:
            raise HTTPException(status_code=503, detail="RAG Pipeline connection failed")
        
        # Get system stats if available
        stats = {}
        if hasattr(rag_pipeline, 'get_performance_stats'):
            stats = rag_pipeline.get_performance_stats()
        
        return {
            "status": "healthy",
            "rag_pipeline": "connected",
            "database": "connected",
            "cache_enabled": hasattr(rag_pipeline, 'semantic_cache'),
            "optimized": hasattr(rag_pipeline, 'ask_question_async'),
            "stats": stats
        }
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Enhanced endpoint untuk mengajukan pertanyaan dengan dukungan async dan caching"""
    if not app_startup_complete:
        raise HTTPException(status_code=503, detail="Application is still starting up")
    
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Use async method if available
        if hasattr(rag_pipeline, 'ask_question_async'):
            result = await rag_pipeline.ask_question_async(
                question=request.question,
                metadata_filter=request.metadata_filter,
                use_cache=request.use_cache
            )
        else:
            # Fallback to synchronous method
            result = rag_pipeline.ask_question(request.question)
        
        return QuestionResponse(
            answer=result["answer"],
            source_urls=result.get("source_urls", []),
            status=result.get("status", "success"),
            source_count=result.get("source_count", 0),
            response_time=result.get("response_time"),
            cached=result.get("cached", False),
            cache_type=result.get("cache_type")
        )
        
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ask/batch", response_model=BatchQuestionResponse)
async def ask_questions_batch(request: BatchQuestionRequest):
    """Endpoint untuk mengajukan multiple pertanyaan secara batch"""
    if not app_startup_complete:
        raise HTTPException(status_code=503, detail="Application is still starting up")
    
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    if not request.questions:
        raise HTTPException(status_code=400, detail="Questions list cannot be empty")
    
    if len(request.questions) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 questions per batch")
    
    try:
        import time
        start_time = time.time()
        
        # Use batch processing if available (optimized pipeline)
        if hasattr(rag_pipeline, 'batch_questions'):
            results = await rag_pipeline.batch_questions(request.questions)
        else:
            # Fallback to sequential processing
            results = []
            for question in request.questions:
                result = rag_pipeline.ask_question(question)
                results.append(result)
        
        processing_time = time.time() - start_time
        
        # Convert results to response format
        response_results = []
        for result in results:
            response_results.append(QuestionResponse(
                answer=result["answer"],
                source_urls=result.get("source_urls", []),
                status=result.get("status", "success"),
                source_count=result.get("source_count", 0),
                response_time=result.get("response_time"),
                cached=result.get("cached", False),
                cache_type=result.get("cache_type")
            ))
        
        return BatchQuestionResponse(
            results=response_results,
            total_questions=len(request.questions),
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        logging.error(f"Error processing batch questions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get comprehensive system statistics"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        base_stats = {
            "embedding_model": os.getenv("EMBEDDING_MODEL"),
            "llm_model": os.getenv("LLM_MODEL"),
            "chunk_size": int(os.getenv("CHUNK_SIZE", 500)),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 50)),
        }
          # Get enhanced stats from optimized pipeline
        if hasattr(rag_pipeline, 'get_performance_stats'):
            enhanced_stats = rag_pipeline.get_performance_stats()
            base_stats.update(enhanced_stats)
        
        return base_stats
        
    except Exception as e:
        logging.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/cache/clear")
async def clear_cache():
    """Clear semantic cache"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        if hasattr(rag_pipeline, 'semantic_cache') and rag_pipeline.semantic_cache:
            rag_pipeline.semantic_cache.clear_cache()
            return {"message": "Cache cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Cache not available in current pipeline")
            
    except Exception as e:
        logging.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        if hasattr(rag_pipeline, 'semantic_cache') and rag_pipeline.semantic_cache:
            stats = rag_pipeline.semantic_cache.export_cache_stats()
            return stats
        else:
            return {"message": "Cache not available in current pipeline"}
            
    except Exception as e:
        logging.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

# Example questions endpoint
@app.get("/examples")
async def get_example_questions():
    """Get example questions that can be asked"""
    return {
        "example_questions": [
            "Apa itu Universitas Gunadarma?",
            "Fakultas apa saja yang ada di Universitas Gunadarma?",
            "Bagaimana cara mendaftar di Universitas Gunadarma?",
            "Apa saja program studi yang tersedia?",
            "Dimana lokasi kampus Universitas Gunadarma?",
            "Bagaimana sistem pembelajaran di Universitas Gunadarma?",
            "Apa saja fasilitas yang tersedia di kampus?",
            "Bagaimana cara menghubungi BAAK Universitas Gunadarma?"
        ]
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )