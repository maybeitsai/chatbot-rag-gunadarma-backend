from typing import Optional, List, Dict, Any
import os
from app.rag.data_cleaner import DataCleaner
from langchain.schema import Document
from rich.console import Console
from rich.status import Status
import logging
from scripts.src.setup.core.tracker import StepTracker, SetupStep

class DataProcessor:
    """Data processing and optimization with Rich display"""
    
    def __init__(self, logger: logging.Logger, tracker: StepTracker):
        self.logger = logger
        self.tracker = tracker
        self.console = Console()
    
    def process_and_optimize_data(self) -> bool:
        """Process and optimize data with cleaning"""
        try:
            data_file = self._find_data_file()
            if not data_file:
                self.tracker.log_step(SetupStep.DATA_PROCESS, False, "No data file found")
                return False
            
            with Status("🧹 Cleaning data...", console=self.console):
                cleaned_data = self._clean_data(data_file)
            
            with Status("📄 Converting to documents...", console=self.console):
                documents = self._convert_to_documents(cleaned_data)
            
            if not documents:
                self.tracker.log_step(SetupStep.DATA_PROCESS, False, "No valid documents")
                return False
            
            return self._setup_vector_store(documents)
            
        except Exception as e:
            self.tracker.log_step(SetupStep.DATA_PROCESS, False, str(e))
            return False
    
    def _find_data_file(self) -> Optional[str]:
        """Find available data file"""
        for file_path in ["data/output.json", "data/output.csv"]:
            if os.path.exists(file_path):
                return file_path
        return None
    
    def _clean_data(self, data_file: str) -> List[Dict[str, Any]]:
        """Clean data from file"""
        cleaner = DataCleaner()
        
        if hasattr(cleaner, 'clean_data_from_file'):
            return cleaner.clean_data_from_file(data_file)
        elif hasattr(cleaner, 'clean_and_deduplicate_data'):
            import json
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f) if data_file.endswith('.json') else []
            return cleaner.clean_and_deduplicate_data(raw_data)
        else:
            from app.rag.data_cleaner import clean_data_file
            import tempfile
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                tmp_output = tmp_file.name
            
            clean_data_file(data_file, tmp_output)
            
            with open(tmp_output, 'r', encoding='utf-8') as f:
                cleaned_data = json.load(f)
            
            os.unlink(tmp_output)
            return cleaned_data
    
    def _convert_to_documents(self, cleaned_data: List[Dict[str, Any]]) -> List:
        """Convert cleaned data to Document objects"""
        documents = []
        self.console.print(f"📄 Converting {len(cleaned_data)} items to Document objects...")
        
        for i, item in enumerate(cleaned_data):
            if not isinstance(item, dict):
                continue
            
            content = (item.get('text_content') or item.get('content') or 
                      item.get('text') or item.get('page_content') or '')
            
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
        
        self.tracker.log_step(
            SetupStep.DATA_PROCESS, 
            True, 
            f"Converted {len(documents)} documents"
        )
        return documents
    
    def _setup_vector_store(self, documents: List) -> bool:
        """Setup vector store with documents"""
        from app.rag.vector_store import VectorStoreManager
        
        try:
            with Status("⚡ Setting up vector store...", console=self.console):
                vector_manager = VectorStoreManager()
                vector_manager.setup_and_populate_from_documents(documents)
            
            try:
                with Status("🔧 Creating performance indexes...", console=self.console):
                    vector_manager.create_indexes()
                self.console.print("✅ Performance indexes created", style="green")
            except Exception as index_error:
                self.console.print(f"⚠️ Index creation failed (non-critical): {index_error}", style="yellow")
            
            return True
            
        except Exception as e:
            self.tracker.log_step(SetupStep.DATA_PROCESS, False, str(e))
            return False