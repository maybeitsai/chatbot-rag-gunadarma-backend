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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("setup.log"),
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
    
    def setup_database(self) -> bool:
        """Setup database and tables"""
        logger.info("Setting up database...")
        
        try:
            from rag.db_setup import setup_database
            setup_database()
            self.log_step("Database Setup", True, "Database tables created successfully")
            return True
        except Exception as e:
            self.log_step("Database Setup", False, str(e))
            return False
    
    def crawl_data(self) -> bool:
        """Crawl data if needed"""
        logger.info("Checking data availability...")
        data_files = ["data/output.json", "data/output.csv"]
        existing_files = [f for f in data_files if os.path.exists(f)]
        
        if existing_files:
            self.log_step(
                "Data Check", 
                True, 
                f"Found existing data files: {', '.join(existing_files)}"
            )
            return True
        
        logger.info("No existing data found, starting crawling process...")
        try:
            from crawl.crawler import crawl_pipeline
            crawl_pipeline()
            self.log_step("Data Crawling", True, "Successfully crawled data")
            return True
        except Exception as e:
            self.log_step("Data Crawling", False, str(e))
            return False
    
    def process_and_optimize_data(self) -> bool:
        """Process and optimize data with cleaning"""
        logger.info("Processing and optimizing data...")
        
        try:
            from rag.data_cleaner import DataCleaner
            from rag.vector_store import VectorStoreManager
            
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
                from rag.data_cleaner import clean_data_file
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
            
            # Setup vector store
            vector_manager = VectorStoreManager()
            vector_manager.setup_and_populate_from_documents(cleaned_data)
            self.log_step(
                "Vector Store Setup", 
                True, 
                "Vector store created and populated"
            )
            
            return True
            
        except Exception as e:
            self.log_step("Data Processing", False, str(e))
            return False
    
    def test_system(self) -> bool:
        """Test the complete system"""
        logger.info("Testing system functionality...")
        
        try:
            from rag.pipeline import create_rag_pipeline
            
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
    
    async def run_complete_setup(self, skip_crawling: bool = False) -> bool:
        """Run complete system setup"""
        logger.info("Starting complete RAG system setup...")
        logger.info("=" * 60)
        
        # Step 1: Check environment
        if not self.check_environment():
            return False
        
        # Step 2: Setup database
        if not self.setup_database():
            return False
        
        # Step 3: Crawl data (if needed)
        if not skip_crawling and not self.crawl_data():
            return False
        
        # Step 4: Process and optimize data
        if not self.process_and_optimize_data():
            return False
        
        # Step 5: Test system
        if not self.test_system():
            return False
        
        logger.info("=" * 60)
        logger.info("[SUCCESS] RAG SYSTEM SETUP COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("System is ready! You can now:")
        logger.info("   Start the API: uv run uvicorn main:app --reload")
        logger.info("   Run tests: uv run python test/test_system.py")
        logger.info("   Check performance: uv run python test/test_performance.py")
        logger.info("   Run optimization: uv run python optimize.py")
        
        return True


async def main():
    """Main setup function"""
    setup = RAGSystemSetup()
    
    # Parse command line arguments
    skip_crawling = "--skip-crawling" in sys.argv
    
    try:
        success = await setup.run_complete_setup(skip_crawling=skip_crawling)
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\nSetup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Setup failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)