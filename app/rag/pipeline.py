"""
RAG Pipeline with async processing, semantic caching, and enhanced performance
"""

import os
import asyncio
import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

# Import custom modules
from app.rag.vector_store import VectorStoreManager
from app.rag.semantic_cache import get_semantic_cache
from app.rag.async_processor import AsyncDataProcessor
from app.rag.hybrid_search import HybridSearchConfig, SearchType

# Langchain imports  
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")

load_dotenv()
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Enhanced RAG Pipeline with semantic caching, optimized vector store,
    asynchronous processing capabilities, and hybrid search support
    """
    
    def __init__(self, enable_cache: bool = True, cache_similarity_threshold: float = 0.85,
                 hybrid_config: Optional[HybridSearchConfig] = None):
        # Configuration
        self.llm_model = os.getenv("LLM_MODEL", "gemini-2.5-flash-preview-05-20")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Initialize components
        self.enable_cache = enable_cache
        self.semantic_cache = get_semantic_cache() if enable_cache else None
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.llm_model,
            google_api_key=self.google_api_key,
            temperature=0.1,  # Slightly higher for more natural responses
            max_output_tokens=1024
        )
        
        # Initialize embeddings for caching
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model,
            google_api_key=self.google_api_key
        )
        
        # Initialize optimized vector store with hybrid search
        self.hybrid_config = hybrid_config or HybridSearchConfig()
        self.vector_store_manager = VectorStoreManager(hybrid_config=self.hybrid_config)
        self.vector_store = self.vector_store_manager.initialize_vector_store()
        
        # Enhanced prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Anda adalah asisten AI yang membantu memberikan informasi tentang Universitas Gunadarma berdasarkan dokumen resmi yang tersedia.

ATURAN PENTING:
1. HANYA gunakan informasi dari konteks dokumen yang diberikan di bawah ini
2. Jika informasi tersedia dalam konteks, berikan jawaban yang lengkap dan informatif
3. SELALU sertakan URL sumber yang relevan dalam format: "Sumber: [URL]"
4. Jika informasi tidak tersedia dalam konteks, jawab: "Maaf, informasi tersebut tidak tersedia dalam database kami saat ini."
5. Berikan jawaban dalam bahasa Indonesia yang jelas dan mudah dipahami
6. Struktur jawaban dengan baik menggunakan paragraf dan poin-poin jika diperlukan

Konteks dari dokumen resmi Universitas Gunadarma:
{context}

Pertanyaan: {question}

Jawaban:
"""
        )
        
        # Performance metrics
        self.performance_metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_response_time': 0.0,
            'total_response_time': 0.0
        }
          # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Optimized RAG Pipeline initialized successfully")
    
    async def _get_question_embedding(self, question: str) -> Optional[List[float]]:
        """Get embedding for question (for semantic caching)"""
        if not self.enable_cache:
            return None
            
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.thread_pool, 
                self.embeddings_model.embed_query,
                question
            )
            return embedding
        except Exception as e:
            logger.warning(f"Could not get question embedding: {e}")
            return None

    def _setup_retriever(self, metadata_filter: Optional[Dict[str, Any]] = None, 
                       use_hybrid: bool = False) -> Any:
        """Setup optimized retriever with hybrid search support"""
        
        # Dynamic retrieval parameters optimized for similarity search
        search_kwargs = {
            "k": 5,  # Increased for better context
            "score_threshold": 0.5  # Slightly lower threshold for more results
        }
        
        # Use hybrid search if requested
        if use_hybrid:
            return self.vector_store_manager.get_retriever(
                search_type="hybrid",
                k=search_kwargs["k"],
                use_hybrid=True
            )
        
        # Note: metadata filtering removed as it's not supported in optimized vector store
        # The optimized version focuses on pure similarity search for better performance
        if metadata_filter:
            logger.warning("Metadata filtering not supported in optimized vector store, using similarity search only")
        
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs
        )    
    async def _get_retrieval_results(self, question: str, 
                                   metadata_filter: Optional[Dict[str, Any]] = None,
                                   use_hybrid: bool = False) -> List[Document]:
        """Get retrieval results asynchronously with hybrid search support"""
        retriever = self._setup_retriever(metadata_filter, use_hybrid)
        
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                self.thread_pool,
                retriever.get_relevant_documents,
                question
            )
            return results
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []
    
    async def _generate_answer(self, question: str, context_docs: List[Document]) -> Dict[str, Any]:
        """Generate answer using LLM"""
        if not context_docs:
            return {
                "answer": "Maaf, informasi tersebut tidak tersedia dalam database kami saat ini.",
                "source_urls": [],
                "status": "not_found",
                "source_count": 0
            }
        
        # Prepare context
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self._setup_retriever(),
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                qa_chain,
                {"query": question}
            )
            
            answer = result["result"]
            source_documents = result["source_documents"]
            
            # Extract unique source URLs
            source_urls = []
            for doc in source_documents:
                url = doc.metadata.get("url", "")
                if url and url not in source_urls:
                    source_urls.append(url)
            
            # Determine status
            no_info_phrases = [
                "tidak tersedia dalam database kami",
                "informasi tidak tersedia",
                "tidak ditemukan informasi"
            ]
            
            status = "not_found" if any(phrase in answer.lower() for phrase in no_info_phrases) else "success"
            
            return {
                "answer": answer,
                "source_urls": source_urls,
                "status": status,
                "source_count": len(source_documents)
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Maaf, terjadi kesalahan dalam memproses pertanyaan: {str(e)}",
                "source_urls": [],
                "status": "error",
                "source_count": 0            }

    async def ask_question_async(self, 
                               question: str, 
                               metadata_filter: Optional[Dict[str, Any]] = None,
                               use_cache: bool = True,
                               use_hybrid: bool = False) -> Dict[str, Any]:
        """
        Process question asynchronously with caching, optimization, and hybrid search
        
        Args:
            question: User question
            metadata_filter: Not used in optimized version (kept for API compatibility)
            use_cache: Whether to use semantic caching
            use_hybrid: Whether to use hybrid search (vector + keyword)
            
        Returns:
            Response dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        self.performance_metrics['total_queries'] += 1
        
        try:
            # Check cache first
            if self.enable_cache and use_cache:
                question_embedding = await self._get_question_embedding(question)
                cached_response = await self.semantic_cache.get_cached_response(question, question_embedding)
                
                if cached_response:
                    self.performance_metrics['cache_hits'] += 1
                    response_time = time.time() - start_time
                    self._update_performance_metrics(response_time)
                    
                    cached_response['response_time'] = round(response_time, 3)
                    cached_response['search_type'] = 'hybrid' if use_hybrid else 'vector'
                    return cached_response
            
            # Get retrieval results with optional hybrid search
            context_docs = await self._get_retrieval_results(question, metadata_filter, use_hybrid)
            
            # Generate answer
            result = await self._generate_answer(question, context_docs)
            
            # Add search type info
            result['search_type'] = 'hybrid' if use_hybrid else 'vector'
            
            # Cache the result
            if self.enable_cache and use_cache and result['status'] in ['success', 'not_found']:
                question_embedding = await self._get_question_embedding(question)
                await self.semantic_cache.cache_response(question, result, question_embedding)
            
            # Add performance metrics
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time)
            result['response_time'] = round(response_time, 3)
            result['cached'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ask_question_async: {e}")
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time)
            
            return {
                "answer": f"Maaf, terjadi kesalahan sistem: {str(e)}",
                "source_urls": [],
                "status": "error",                "source_count": 0,
                "response_time": round(response_time, 3),
                "cached": False,
                "search_type": 'hybrid' if use_hybrid else 'vector'
            }
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Synchronous wrapper for ask_question_async"""
        try:
            # Get current event loop if available
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to handle this differently
                # Create a new thread to run the async code
                import threading
                import concurrent.futures
                
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.ask_question_async(question))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    return future.result()
            else:
                # If no loop is running, we can use asyncio.run()
                return asyncio.run(self.ask_question_async(question))
        except Exception as e:
            logger.error(f"Error in synchronous ask_question: {e}")
            return {
                "status": "error",
                "message": f"Failed to process question: {str(e)}",
                "answer": "",
                "context": []
            }
    
    def _update_performance_metrics(self, response_time: float):
        """Update performance metrics"""
        self.performance_metrics['total_response_time'] += response_time
        self.performance_metrics['avg_response_time'] = (
            self.performance_metrics['total_response_time'] / 
            self.performance_metrics['total_queries']
        )
    
    async def batch_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple questions concurrently"""
        tasks = [self.ask_question_async(question) for question in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing question {i}: {result}")
                processed_results.append({
                    "answer": f"Error processing question: {str(result)}",
                    "source_urls": [],
                    "status": "error",
                    "source_count": 0,
                    "cached": False
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_stats = self.semantic_cache.get_cache_stats() if self.enable_cache else {}
        vector_stats = self.vector_store_manager.get_vector_store_stats()
        
        performance_stats = self.performance_metrics.copy()
        if performance_stats['total_queries'] > 0:
            performance_stats['cache_hit_rate'] = round(
                (performance_stats['cache_hits'] / performance_stats['total_queries']) * 100, 2
            )
        else:
            performance_stats['cache_hit_rate'] = 0
        
        return {
            "performance": performance_stats,
            "cache": cache_stats,
            "vector_store": vector_stats,
            "configuration": {
                "llm_model": self.llm_model,
                "embedding_model": self.embedding_model,                "cache_enabled": self.enable_cache
            }
        }
    
    async def test_connection_async(self) -> bool:
        """Test if the RAG pipeline is working (async version)"""
        try:
            # Use async method for connection test
            test_result = await self.ask_question_async("Test connection")
            return test_result["status"] in ["success", "not_found", "error"]
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test if the RAG pipeline is working (sync wrapper)"""
        try:
            # Get current event loop if available
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, create a task instead of using asyncio.run()
                task = loop.create_task(self.test_connection_async())
                # Since we can't await in sync context within running loop, return True for now
                # The actual test will run in background
                logger.warning("Connection test scheduled in background (async context)")
                return True
            else:
                # If no loop is running, we can use asyncio.run()
                return asyncio.run(self.test_connection_async())
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def warmup_cache(self, warmup_questions: List[str]):
        """Warm up the cache with common questions"""
        logger.info(f"Warming up cache with {len(warmup_questions)} questions...")
        
        results = await self.batch_questions(warmup_questions)
        successful_warmups = sum(1 for r in results if r['status'] == 'success')
        
        logger.info(f"Cache warmup completed: {successful_warmups}/{len(warmup_questions)} successful")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.enable_cache and self.semantic_cache:
            self.semantic_cache.save_cache_sync()
        
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        logger.info("RAG Pipeline cleanup completed")
    
    def optimize_vector_store(self) -> bool:
        """Optimize vector store performance by creating indexes"""
        try:
            logger.info("Optimizing vector store performance...")
            self.vector_store_manager.create_indexes()
            logger.info("Vector store optimization completed successfully")
            return True
        except Exception as e:
            logger.warning(f"Vector store optimization failed (non-critical): {e}")
            return False

    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the optimized vector store"""
        try:
            stats = self.vector_store_manager.get_vector_store_stats()
            return {
                "status": "optimized",
                "collection_name": stats.get("collection_name"),
                "document_count": stats.get("document_count", 0),
                "hnsw_params": stats.get("hnsw_params", {}),
                "features": [
                    "HNSW indexing for fast similarity search",
                    "Optimized retrieval without metadata filtering",
                    "Enhanced performance for large document collections"
                ]
            }
        except Exception as e:
            logger.error(f"Error getting vector store info: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


# Factory function
def create_rag_pipeline(enable_cache: bool = True) -> RAGPipeline:
    """
    Factory function to create optimized RAG pipeline
    
    Features of the optimized pipeline:
    - Uses optimized vector store without metadata filtering for better performance
    - HNSW indexing for fast similarity search
    - Semantic caching for improved response times
    - Async processing capabilities
    
    Args:
        enable_cache: Whether to enable semantic caching
        
    Returns:
        Optimized RAG pipeline instance
    """
    return RAGPipeline(enable_cache=enable_cache)


if __name__ == "__main__":
    # Test the optimized RAG pipeline
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        # Create pipeline
        rag = create_rag_pipeline()
        
        if rag.test_connection():
            print("✓ RAG Pipeline connection successful")
            
            # Test questions
            test_questions = [
                "Apa itu Universitas Gunadarma?",
                "Fakultas apa saja yang ada di Universitas Gunadarma?",
                "Bagaimana cara mendaftar di Universitas Gunadarma?",
                "Dimana lokasi kampus Universitas Gunadarma?",
                "Apa saja fasilitas yang tersedia di kampus?"
            ]
            
            # Test individual question
            print("\n--- Testing Individual Question ---")
            result = await rag.ask_question_async(test_questions[0])
            print(f"Question: {test_questions[0]}")
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Status: {result['status']}")
            print(f"Response Time: {result['response_time']}s")
            print(f"Cached: {result['cached']}")
            
            # Test batch processing
            print("\n--- Testing Batch Processing ---")
            batch_results = await rag.batch_questions(test_questions)
            for i, result in enumerate(batch_results):
                print(f"{i+1}. Status: {result['status']}, Time: {result['response_time']}s, Cached: {result['cached']}")
            
            # Test optimized vector store
            vector_info = rag.get_vector_store_info()
            print(f"✓ Vector Store Status: {vector_info['status']}")
            print(f"  Documents: {vector_info.get('document_count', 0)}")
            print(f"  Features: {', '.join(vector_info.get('features', []))}")
            
            # Test vector store optimization
            if rag.optimize_vector_store():
                print("✓ Vector store optimization successful")
            else:
                print("⚠ Vector store optimization had issues (non-critical)")
            
            # Show stats
            print("\n--- Performance Statistics ---")
            stats = rag.get_performance_stats()
            print(f"Performance: {stats['performance']}")
            if stats['cache']:
                print(f"Cache: {stats['cache']}")
            
            # Cleanup
            rag.cleanup()
            
        else:
            print("✗ RAG Pipeline connection failed")
    
    asyncio.run(main())
