"""
Business logic services untuk RAG operations
"""
import logging
import time
from typing import List, Dict, Any, Optional

from app.api.models import QuestionResponse
from app.api.exceptions import InternalServerError

logger = logging.getLogger(__name__)


class RAGService:
    """Service untuk operasi RAG"""
    
    @staticmethod
    async def process_question(
        rag_pipeline, 
        question: str, 
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> QuestionResponse:
        """Process single question"""
        try:
            # Use async method if available
            if hasattr(rag_pipeline, 'ask_question_async'):
                result = await rag_pipeline.ask_question_async(
                    question=question,
                    metadata_filter=metadata_filter,
                    use_cache=use_cache
                )
            else:
                # Fallback to synchronous method
                result = rag_pipeline.ask_question(question)
            
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
            logger.error(f"Error processing question: {e}")
            raise InternalServerError(str(e))
    
    @staticmethod
    async def process_batch_questions(
        rag_pipeline, 
        questions: List[str], 
        use_cache: bool = True
    ) -> tuple[List[QuestionResponse], float]:
        """Process batch questions"""
        try:
            start_time = time.time()
            
            # Use batch processing if available (optimized pipeline)
            if hasattr(rag_pipeline, 'batch_questions'):
                results = await rag_pipeline.batch_questions(questions)
            else:
                # Fallback to sequential processing
                results = []
                for question in questions:
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
            
            return response_results, round(processing_time, 3)
            
        except Exception as e:
            logger.error(f"Error processing batch questions: {e}")
            raise InternalServerError(str(e))


class HealthService:
    """Service untuk health check operations"""
    
    @staticmethod
    def check_rag_pipeline_health(rag_pipeline) -> Dict[str, Any]:
        """Check RAG pipeline health"""
        try:
            is_healthy = rag_pipeline.test_connection()
            
            if not is_healthy:
                raise Exception("RAG Pipeline connection failed")
            
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
            logger.error(f"Health check failed: {e}")
            raise InternalServerError(f"Health check failed: {str(e)}")


class StatsService:
    """Service untuk system statistics"""
    
    @staticmethod
    def get_system_stats(rag_pipeline) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        import os
        
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
            logger.error(f"Error getting stats: {e}")
            raise InternalServerError(f"Failed to get stats: {str(e)}")


class CacheService:
    """Service untuk cache operations"""
    
    @staticmethod
    def clear_cache(rag_pipeline) -> Dict[str, str]:
        """Clear semantic cache"""
        try:
            if hasattr(rag_pipeline, 'semantic_cache') and rag_pipeline.semantic_cache:
                rag_pipeline.semantic_cache.clear_cache()
                return {"message": "Cache cleared successfully"}
            else:
                from app.api.exceptions import CacheNotAvailableError
                raise CacheNotAvailableError()
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise InternalServerError(f"Failed to clear cache: {str(e)}")
    
    @staticmethod
    def get_cache_stats(rag_pipeline) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if hasattr(rag_pipeline, 'semantic_cache') and rag_pipeline.semantic_cache:
                stats = rag_pipeline.semantic_cache.export_cache_stats()
                return stats
            else:
                return {"message": "Cache not available in current pipeline"}
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            raise InternalServerError(f"Failed to get cache stats: {str(e)}")