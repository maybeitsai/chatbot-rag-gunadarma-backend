#!/usr/bin/env python3
"""
Unified Setup Script for RAG System
Handles complete system initialization, data processing, and optimization
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import time
from dotenv import load_dotenv

# Add project root to path
script_dir = Path(__file__).parent
backend_dir = script_dir.parent
sys.path.insert(0, str(backend_dir))

# Load environment variables from .env file
load_dotenv(backend_dir / '.env')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/setup.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RAGSystemSetup:
    """Unified RAG System Setup and Optimization"""
    
    def __init__(self):
        self.setup_start_time = time.time()
        self.steps_completed = []
        self.errors = []
        
    def log_step(self, step_name: str, success: bool = True, details: str = ""):
        """Log setup step completion"""
        status = "[OK]" if success else "[FAIL]"
        logger.info(f"{status} {step_name}")
        
        if details:
            logger.info(f"   {details}")
            
        if success:
            self.steps_completed.append(step_name)
        else:
            self.errors.append(f"{step_name}: {details}")
    
    def check_environment(self) -> bool:
        """Check if environment is properly configured"""
        logger.info("Checking environment configuration...")
        
        required_env_vars = [
            "GOOGLE_API_KEY",
            "NEON_CONNECTION_STRING",
            "LLM_MODEL",
            "EMBEDDING_MODEL"
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.log_step(
                "Environment Check", 
                False, 
                f"Missing environment variables: {', '.join(missing_vars)}"
            )
            return False
        
        self.log_step("Environment Check", True, "All required environment variables found")
        return True
    
    def manage_cache(self, action: str = "status") -> bool:
        """Manage crawler cache system"""
        logger.info(f"Cache management: {action}")
        
        try:
            from app.crawl.config import CrawlConfig
            from app.crawl.cache_manager import CacheManager
            
            config = CrawlConfig()
            cache_manager = CacheManager(config)
            
            if action == "status":
                stats = cache_manager.get_cache_statistics()
                
                details = [
                    f"Total requests: {stats['total_requests']}",
                    f"Hit rate: {stats['hit_rate']:.1f}%",
                    f"Cache entries: {sum(stats['cache_sizes'].values())}"
                ]
                
                # Show individual cache sizes
                cache_details = []
                for cache_type, size in stats['cache_sizes'].items():
                    if size > 0:
                        cache_details.append(f"{cache_type}: {size}")
                
                if cache_details:
                    details.append(f"({', '.join(cache_details)})")
                
                self.log_step("Cache Status", True, "; ".join(details))
                
            elif action == "cleanup":
                cache_manager.cleanup_expired_cache()
                self.log_step("Cache Cleanup", True, "Expired cache entries removed")
                
            elif action == "clear":
                # Clear all cache
                cache_manager.url_cache.clear()
                cache_manager.content_cache.clear()
                cache_manager.response_cache.clear()
                cache_manager.similarity_cache.clear()
                cache_manager.robots_cache.clear()
                
                # Also clear persistent cache files
                cache_dir = Path("cache")
                if cache_dir.exists():
                    cache_files = ["url_cache.json", "response_cache.json", 
                                 "similarity_cache.json", "robots_cache.json"]
                    for cache_file in cache_files:
                        cache_path = cache_dir / cache_file
                        if cache_path.exists():
                            cache_path.unlink()
                
                self.log_step("Cache Clear", True, "All cache data cleared")
                
            return True
            
        except ImportError:
            self.log_step(f"Cache {action.title()}", False, "Enhanced cache manager not available")
            return False
        except Exception as e:
            self.log_step(f"Cache {action.title()}", False, str(e))
            return False
    
    def setup_database(self) -> bool:
        """Setup database and tables"""
        logger.info("Setting up database...")
        
        try:
            from app.rag.db_setup import setup_database
            setup_database()
            self.log_step("Database Setup", True, "Database tables created successfully")
            return True
        except Exception as e:
            self.log_step("Database Setup", False, str(e))
            return False
    
    async def crawl_data(self, force_crawl: bool = False) -> bool:
        """Enhanced crawl data using optimized crawler with caching"""
        logger.info("Checking data availability...")
        data_files = ["data/output.json", "data/output.csv"]
        existing_files = [f for f in data_files if os.path.exists(f)]
        
        if existing_files and not force_crawl:
            self.log_step(
                "Data Check", 
                True, 
                f"Found existing data files: {', '.join(existing_files)}"
            )
            
            # Ask user if they want to recrawl with enhanced features
            response = input("\nExisting data found. Recrawl with enhanced caching? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                logger.info("Skipping crawling, using existing data...")
                return True
        elif force_crawl:
            logger.info("Force crawl mode: Recrawling regardless of existing data...")
        
        logger.info("Starting enhanced crawling with advanced caching...")
        try:
            from app.crawl.config import CrawlConfig
            from app.crawl.crawler import WebCrawler
            
            # Configure enhanced crawler with all caching features
            config = CrawlConfig(
                max_depth=0,
                similarity_threshold=0.8,
                duplicate_threshold=0.95,
                request_delay=1.0,
                baak_delay=2.0,
                
                # Enable all caching features
                enable_url_cache=True,
                enable_content_cache=True,
                enable_response_cache=True,
                enable_smart_filtering=True,
                enable_robots_respect=True,
                
                # Cache configuration
                cache_ttl=3600,  # 1 hour
                max_cache_size=1000,
                
                # Performance settings
                max_retries=3,
                timeout=60
            )
            
            # Target URLs for Gunadarma
            target_urls = [
                "https://baak.gunadarma.ac.id/", "https://www.gunadarma.ac.id/"
            ]
            
            logger.info(f"[ENHANCED CRAWLING CONFIGURATION]")
            logger.info(f"   Target URLs: {len(target_urls)}")
            logger.info(f"   Caching: URL[YES] Content[YES] Response[YES]")
            logger.info(f"   Smart filtering: [YES]")
            logger.info(f"   Robots.txt respect: [YES]")
            logger.info(f"   Duplicate threshold: {config.duplicate_threshold}")
            logger.info(f"   Max depth: {config.max_depth}")
            
            # Initialize and run crawler
            crawler = WebCrawler(target_urls, config)
            report = await crawler.crawl(incremental=True)
            
            if report.get('status') == 'success':
                summary = report['crawl_summary']
                cache_stats = report.get('cache_statistics', {})
                
                # Log detailed results
                details = [
                    f"Duration: {summary.get('duration_seconds', 0):.1f}s",
                    f"Pages crawled: {summary.get('total_pages_crawled', 0)}",
                    f"PDFs processed: {summary.get('total_pdfs_processed', 0)}",
                    f"Pages updated: {summary.get('pages_updated', 0)}",
                    f"Pages skipped: {summary.get('pages_skipped', 0)}",
                    f"Duplicates skipped: {summary.get('duplicates_skipped', 0)}",
                    f"Cache hits: {summary.get('cache_hits', 0)}",
                    f"Cache hit rate: {cache_stats.get('hit_rate', 0):.1f}%",
                    f"Total saved: {summary.get('total_saved', 0)}"
                ]
                
                self.log_step(
                    "Enhanced Data Crawling", 
                    True, 
                    "; ".join(details)
                )
                
                # Log efficiency gains
                if summary.get('cache_hits', 0) > 0 or summary.get('duplicates_skipped', 0) > 0:
                    efficiency_details = []
                    if summary.get('cache_hits', 0) > 0:
                        saved_time = summary['cache_hits'] * config.request_delay
                        efficiency_details.append(f"Cache saved ~{saved_time:.1f}s")
                    if summary.get('duplicates_skipped', 0) > 0:
                        efficiency_details.append(f"Prevented {summary['duplicates_skipped']} redundant crawls")
                    
                    self.log_step(
                        "Crawling Efficiency", 
                        True, 
                        "; ".join(efficiency_details)
                    )
                return True
            else:
                error_msg = report.get('error', 'Unknown crawling error')
                self.log_step("Enhanced Data Crawling", False, error_msg)
                return False
                
        except ImportError as e:
            logger.warning(f"Enhanced crawler not available: {e}")
            logger.info("Falling back to basic crawler...")
            # Fallback to basic crawler
            try:
                from app.crawl.crawler import crawl_pipeline
                await crawl_pipeline()
                self.log_step("Basic Data Crawling", True, "Successfully crawled data with basic crawler")
                return True
            except Exception as e:
                self.log_step("Basic Data Crawling", False, str(e))
                return False
                
        except Exception as e:
            logger.error(f"Enhanced crawling failed: {e}")
            logger.info("Falling back to basic crawler...")
            # Fallback to basic crawler
            try:
                from app.crawl.crawler import crawl_pipeline
                await crawl_pipeline()
                self.log_step("Fallback Data Crawling", True, "Successfully crawled data with basic crawler")
                return True
            except Exception as fallback_e:
                self.log_step("Data Crawling", False, f"Enhanced: {e}; Fallback: {fallback_e}")
                return False

    def process_and_optimize_data(self) -> bool:
        """Process and optimize data with cleaning"""
        logger.info("Processing and optimizing data...")
        
        try:
            from app.rag.data_cleaner import DataCleaner
            from app.rag.vector_store import VectorStoreManager
            
            # Clean data
            cleaner = DataCleaner()
            data_file = None
            
            # Find data file
            for file_path in ["data/output.json", "data/output.csv"]:
                if os.path.exists(file_path):
                    data_file = file_path
                    break
            
            if not data_file:
                self.log_step("Data Processing", False, "No data file found")
                return False
              # Clean and process data using the correct method
            if hasattr(cleaner, 'clean_data_from_file'):
                cleaned_data = cleaner.clean_data_from_file(data_file)
            elif hasattr(cleaner, 'clean_and_deduplicate_data'):
                # Load data first, then clean
                import json
                with open(data_file, 'r', encoding='utf-8') as f:
                    if data_file.endswith('.json'):
                        raw_data = json.load(f)
                    else:
                        # Handle CSV or other formats
                        raw_data = []
                
                cleaned_data = cleaner.clean_and_deduplicate_data(raw_data)
            else:
                # Use the standalone function
                from app.rag.data_cleaner import clean_data_file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                    tmp_output = tmp_file.name
                
                clean_data_file(data_file, tmp_output)
                
                # Load cleaned data
                with open(tmp_output, 'r', encoding='utf-8') as f:
                    cleaned_data = json.load(f)
                
                # Clean up temp file
                os.unlink(tmp_output)
            
            self.log_step(
                "Data Cleaning", 
                True, 
                f"Cleaned {len(cleaned_data)} documents"
            )
              # Convert cleaned data to Document objects for vector store
            from langchain.schema import Document
            documents = []
            
            logger.info(f"Converting {len(cleaned_data)} cleaned items to Document objects...")
            if cleaned_data:
                logger.info(f"Sample item keys: {list(cleaned_data[0].keys())}")
            
            for i, item in enumerate(cleaned_data):
                if isinstance(item, dict):
                    # Check various possible content field names
                    content = (item.get('text_content') or 
                              item.get('content') or 
                              item.get('text') or 
                              item.get('page_content') or '')
                    
                    if content and len(content.strip()) > 0:
                        doc = Document(
                            page_content=content,
                            metadata={
                                'url': item.get('url', ''),
                                'source_type': item.get('source_type', 'unknown'),
                                'title': item.get('title', ''),
                                'content_length': len(content)
                            }
                        )
                        documents.append(doc)
                    else:
                        if i < 3:  # Only show first 3 warnings to avoid spam
                            logger.warning(f"Item {i} has no valid content. Keys: {list(item.keys())}")
                else:
                    logger.warning(f"Item {i} is not a dict: {type(item)}")
            
            if not documents:
                self.log_step("Data Conversion", False, "No valid documents to convert")
                return False
            
            self.log_step(
                "Data Conversion", 
                True, 
                f"Converted {len(documents)} documents to Document objects"
            )
            
            # Setup vector store
            vector_manager = VectorStoreManager()
            vector_manager.setup_and_populate_from_documents(documents)
            self.log_step(
                "Vector Store Setup", 
                True, 
                "Vector store created and populated"
            )
            
            # Create optimized indexes for better performance
            logger.info("Creating optimized indexes for vector store...")
            try:
                vector_manager.create_indexes()
                self.log_step(
                    "Index Optimization", 
                    True, 
                    "HNSW and other performance indexes created"
                )
            except Exception as index_error:
                self.log_step(
                    "Index Optimization", 
                    False, 
                    f"Index creation failed (non-critical): {index_error}"
                )
                logger.warning("Vector store will work without optimized indexes, but with slower search")
            
            # Get and log vector store statistics
            try:
                stats = vector_manager.get_vector_store_stats()
                logger.info(f"Vector Store Statistics:")
                logger.info(f"  - Collection: {stats['collection_name']}")
                logger.info(f"  - Documents: {stats['document_count']}")
                logger.info(f"  - HNSW params: {stats['hnsw_params']}")
            except Exception as stats_error:
                logger.warning(f"Could not retrieve vector store stats: {stats_error}")
            
            return True
            
        except Exception as e:
            self.log_step("Data Processing", False, str(e))
            return False
    
    def test_vector_store(self) -> bool:
        """Test vector store functionality specifically"""
        logger.info("Testing vector store functionality...")
        
        try:
            from app.rag.vector_store import VectorStoreManager
            
            # Initialize vector store
            vector_manager = VectorStoreManager()
            
            # Test vector store stats
            stats = vector_manager.get_vector_store_stats()
            if stats.get('document_count', 0) == 0:
                self.log_step("Vector Store Test", False, "No documents found in vector store")
                return False
            
            # Test retriever initialization
            retriever = vector_manager.get_retriever(
                search_type="similarity_score_threshold",
                k=3,
                score_threshold=0.3
            )
            
            if not retriever:
                self.log_step("Vector Store Test", False, "Failed to initialize retriever")
                return False
            
            # Test vector store initialization
            vector_store = vector_manager.initialize_vector_store()
            if not vector_store:
                self.log_step("Vector Store Test", False, "Failed to initialize vector store")
                return False
            
            self.log_step(
                "Vector Store Test", 
                True, 
                f"Vector store working with {stats['document_count']} documents"
            )
            return True
            
        except Exception as e:
            self.log_step("Vector Store Test", False, str(e))
            return False
    
    def test_system(self) -> bool:
        """Test the complete system"""
        logger.info("Testing system functionality...")
        
        try:
            from app.rag.pipeline import create_rag_pipeline
            
            # Create pipeline
            pipeline = create_rag_pipeline(enable_cache=True)
            
            # Test connection
            if hasattr(pipeline, 'test_connection'):
                if not pipeline.test_connection():
                    self.log_step("System Test", False, "Pipeline connection test failed")
                    return False
            
            # Test with sample question
            test_question = "Apa itu Universitas Gunadarma?"
            result = pipeline.ask_question(test_question)
            
            if result and result.get('status') in ['success', 'not_found']:
                self.log_step(
                    "System Test", 
                    True, 
                    f"Successfully processed test question: {result.get('status')}"
                )
                return True
            else:
                self.log_step("System Test", False, "Failed to process test question")
                return False
                
        except Exception as e:
            self.log_step("System Test", False, str(e))
            return False
    async def run_complete_setup(self, skip_crawling: bool = False, force_crawl: bool = False) -> bool:
        """Run complete system setup with enhanced options"""
        logger.info("Starting complete RAG system setup...")
        logger.info("=" * 60)
        
        # Step 1: Check environment
        if not self.check_environment():
            return False
        
        # Step 2: Setup database
        if not self.setup_database():
            return False
        
        # Step 3: Cache management
        self.manage_cache("status")
          # Step 4: Enhanced crawl data (if needed)
        if not skip_crawling:
            # Pass force_crawl parameter to crawl_data
            if force_crawl:
                logger.info("Force crawl mode: Will recrawl even if data exists")
            success = await self.crawl_data(force_crawl=force_crawl)
            if not success:
                return False
        else:
            logger.info("Skipping crawling as requested")
          # Step 5: Process and optimize data
        if not self.process_and_optimize_data():
            return False
        
        # Step 6: Test vector store specifically
        if not self.test_vector_store():
            return False
          # Step 7: Test complete system
        if not self.test_system():
            return False
        
        # Step 8: Final optimization
        logger.info("Running final vector store optimization...")
        self.optimize_vector_store()  # Non-critical, so don't fail if this doesn't work
        
        logger.info("=" * 60)
        logger.info("[SUCCESS] RAG SYSTEM SETUP COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("System is ready! You can now:")
        logger.info("   [START] API: uv run uvicorn main:app --reload")
        logger.info("   [TEST] Run tests: uv run python test/test_system.py")
        logger.info("   [PERF] Check performance: uv run python test/test_performance.py") 
        logger.info("   [CACHE] Cache status: python setup.py --cache-status")
        logger.info("   [CRAWL] Recrawl data: python setup.py --crawl-only")
        logger.info("   [OPTIMIZE] Optimize vector store: python setup.py --optimize-vector-store")
        
        return True


    def optimize_vector_store(self) -> bool:
        """Optimize vector store performance"""
        logger.info("Optimizing vector store performance...")
        
        try:
            from app.rag.vector_store import VectorStoreManager
            
            vector_manager = VectorStoreManager()
            
            # Clean up old collections if any
            try:
                vector_manager.cleanup_old_collections(keep_latest=2)
                self.log_step(
                    "Vector Store Cleanup", 
                    True, 
                    "Old collections cleaned up"
                )
            except Exception as cleanup_error:
                self.log_step(
                    "Vector Store Cleanup", 
                    False, 
                    f"Cleanup failed: {cleanup_error}"
                )
            
            # Create optimized indexes
            try:
                vector_manager.create_indexes()
                self.log_step(
                    "Vector Store Optimization", 
                    True, 
                    "Performance indexes created"
                )
            except Exception as index_error:
                self.log_step(
                    "Vector Store Optimization", 
                    False, 
                    f"Index optimization failed: {index_error}"
                )
            
            # Get final stats
            stats = vector_manager.get_vector_store_stats()
            logger.info("Final Vector Store Configuration:")
            logger.info(f"  - Collection: {stats['collection_name']}")
            logger.info(f"  - Total Documents: {stats['document_count']}")
            logger.info(f"  - HNSW Parameters: {stats['hnsw_params']}")
            
            return True
            
        except Exception as e:
            self.log_step("Vector Store Optimization", False, str(e))
            return False

    def show_cache_status(self):
        """Show cache status information"""
        try:
            from app.crawl.cache_manager import CacheManager
            from app.crawl.config import CrawlConfig
            
            config = CrawlConfig()
            cache_manager = CacheManager(config)
            stats = cache_manager.get_cache_statistics()
            
            logger.info("Cache Status:")
            logger.info(f"  Total requests: {stats['total_requests']}")
            logger.info(f"  Hit rate: {stats['hit_rate']:.2f}%")
            logger.info(f"  Cache sizes: {stats['cache_sizes']}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to get cache status: {e}")
            return False
    
    def cleanup_cache(self):
        """Cleanup expired cache entries"""
        try:
            from app.crawl.cache_manager import CacheManager
            from app.crawl.config import CrawlConfig
            
            config = CrawlConfig()
            cache_manager = CacheManager(config)
            cache_manager.cleanup_expired_cache()
            cache_manager.save_cache()
            
            logger.info("Cache cleanup completed")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")
            return False
    
async def main():
    """Enhanced main setup function with comprehensive options"""
    setup = RAGSystemSetup()
      # Parse command line arguments
    args = sys.argv[1:]
    
    # Command line options
    skip_crawling = "--skip-crawling" in args
    force_crawl = "--force-crawl" in args
    cache_only = "--cache-only" in args
    optimize_only = "--optimize-vector-store" in args
    
    # Vector store optimization command
    if optimize_only:
        logger.info("Running vector store optimization only...")
        success = setup.optimize_vector_store()
        return 0 if success else 1
    
    # Cache management commands
    if "--cache-status" in args:
        setup.manage_cache("status")
        return 0
    
    if "--cache-cleanup" in args:
        setup.manage_cache("cleanup")
        return 0
    
    if "--cache-clear" in args:
        setup.manage_cache("clear")
        return 0
    
    # Help command
    if "--help" in args or "-h" in args:
        print_help()
        return 0
    
    # Crawl-only mode
    if "--crawl-only" in args:
        logger.info("Running enhanced crawling only...")
        success = await setup.crawl_data()
        return 0 if success else 1
    
    # Cache-only mode (for testing)
    if cache_only:
        logger.info("Running cache management only...")
        setup.manage_cache("status")
        return 0
      # Show configuration before starting
    logger.info("[RAG SYSTEM SETUP]")
    logger.info("=" * 50)
    logger.info(f"Options:")
    logger.info(f"   Skip crawling: {skip_crawling}")
    logger.info(f"   Force recrawl: {force_crawl}")
    logger.info(f"   Enhanced caching: [YES]")
    logger.info("=" * 50)
    
    try:
        success = await setup.run_complete_setup(
            skip_crawling=skip_crawling,
            force_crawl=force_crawl
        )
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\nSetup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Setup failed with unexpected error: {e}")
        return 1


def print_help():
    """Print help information"""
    help_text = """
RAG SYSTEM SETUP - Enhanced with Advanced Caching
===================================================

USAGE:
    python setup.py [OPTIONS]

MAIN OPTIONS:
    --help, -h             Show this help message
    --skip-crawling        Skip data crawling, use existing data
    --force-crawl          Force recrawl even if data exists
    --crawl-only           Run enhanced crawling only, skip other steps
    --optimize-vector-store Optimize vector store indexes only

CACHE MANAGEMENT:
    --cache-status         Show cache statistics
    --cache-cleanup        Clean up expired cache entries
    --cache-clear          Clear all cache data
    --cache-only           Cache management mode only

EXAMPLES:
    # Complete setup with enhanced crawling
    python setup.py

    # Setup without crawling (use existing data)
    python setup.py --skip-crawling

    # Force recrawl with caching
    python setup.py --force-crawl

    # Run enhanced crawling only
    python setup.py --crawl-only

    # Optimize vector store performance
    python setup.py --optimize-vector-store

    # Check cache status
    python setup.py --cache-status

    # Clean up cache
    python setup.py --cache-cleanup

FEATURES:
    [YES] Enhanced crawler with advanced caching
    [YES] Smart URL filtering and duplicate detection
    [YES] Robots.txt compliance
    [YES] Content similarity caching
    [YES] 4-8x performance improvement
    [YES] Comprehensive analytics and reporting

For more information, see CACHING_IMPLEMENTATION.md
"""
    print(help_text)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)