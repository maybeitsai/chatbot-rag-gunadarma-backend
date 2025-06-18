"""
Asynchronous Data Processor for improved performance
Handles data processing, cleaning, and document creation concurrently
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from rag.data_cleaner import DataCleaner

logger = logging.getLogger(__name__)


class AsyncDataProcessor:
    """
    Asynchronous data processor for handling large datasets efficiently
    """
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 max_workers: int = None,
                 batch_size: int = 100):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize data cleaner
        self.data_cleaner = DataCleaner()
        
        # Thread/Process pool settings
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        
        # Stats
        self.processing_stats = {
            'total_items': 0,
            'processed_items': 0,
            'failed_items': 0,
            'total_documents': 0,
            'processing_time': 0.0
        }
    
    async def load_data_async(self, file_path: str) -> List[Dict]:
        """
        Asynchronously load data from JSON or CSV file
        
        Args:
            file_path: Path to the data file
            
        Returns:
            List of data items
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading data from: {file_path}")
        
        def _load_file():
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_path.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(file_path)
                return df.to_dict('records')
            else:
                raise ValueError("Unsupported file format. Use JSON or CSV.")
        
        # Run file loading in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            data = await loop.run_in_executor(executor, _load_file)
        
        logger.info(f"Loaded {len(data)} items from {file_path}")
        return data
    
    async def clean_data_batch(self, data_batch: List[Dict]) -> List[Dict]:
        """
        Clean a batch of data items asynchronously
        
        Args:
            data_batch: Batch of data items to clean
            
        Returns:
            Cleaned data batch
        """
        loop = asyncio.get_event_loop()
        
        def _clean_batch():
            return self.data_cleaner.clean_and_deduplicate_data(data_batch)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            cleaned_batch = await loop.run_in_executor(executor, _clean_batch)
        
        return cleaned_batch
    
    def _create_document(self, item: Dict) -> List[Document]:
        """
        Create document(s) from a single data item
        
        Args:
            item: Data item with text content and metadata
            
        Returns:
            List of Document objects (may be split into chunks)
        """
        try:
            # Skip items without text content
            if not item.get('text_content') or len(item['text_content'].strip()) < 50:
                return []
            
            # Create metadata
            metadata = {
                'url': item.get('url', ''),
                'title': item.get('title', ''),
                'source_type': item.get('source_type', 'html'),
                'timestamp': item.get('timestamp', ''),
                'content_length': len(item['text_content'])
            }
            
            # Add original URL if it was cleaned
            if item.get('original_url'):
                metadata['original_url'] = item['original_url']
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(item['text_content'])
            
            # Create documents for each chunk
            documents = []
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 20:  # Skip very small chunks
                    continue
                
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = i
                chunk_metadata['total_chunks'] = len(chunks)
                chunk_metadata['chunk_length'] = len(chunk)
                
                doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error creating document from item: {e}")
            return []
    
    async def create_documents_batch(self, data_batch: List[Dict]) -> List[Document]:
        """
        Create documents from a batch of data items asynchronously
        
        Args:
            data_batch: Batch of cleaned data items
            
        Returns:
            List of Document objects
        """
        loop = asyncio.get_event_loop()
        
        # Use ProcessPoolExecutor for CPU-intensive document creation
        with ProcessPoolExecutor(max_workers=min(4, self.max_workers)) as executor:
            # Submit tasks for each item in the batch
            tasks = [
                loop.run_in_executor(executor, self._create_document, item)
                for item in data_batch
            ]
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and filter out exceptions
        documents = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in document creation: {result}")
                self.processing_stats['failed_items'] += 1
            elif isinstance(result, list):
                documents.extend(result)
                self.processing_stats['processed_items'] += 1
        
        return documents
    
    async def process_data_async(self, 
                               file_path: str, 
                               clean_data: bool = True,
                               save_intermediate: bool = True) -> List[Document]:
        """
        Process data file asynchronously with cleaning and document creation
        
        Args:
            file_path: Path to data file
            clean_data: Whether to clean and deduplicate data
            save_intermediate: Whether to save intermediate cleaned data
            
        Returns:
            List of processed Document objects
        """
        start_time = time.time()
        
        try:
            # Load data
            logger.info("Starting async data processing...")
            data = await self.load_data_async(file_path)
            self.processing_stats['total_items'] = len(data)
            
            # Clean data if requested
            if clean_data:
                logger.info("Cleaning and deduplicating data...")
                
                # Process data in batches for memory efficiency
                cleaned_data = []
                for i in range(0, len(data), self.batch_size):
                    batch = data[i:i + self.batch_size]
                    cleaned_batch = await self.clean_data_batch(batch)
                    cleaned_data.extend(cleaned_batch)
                    
                    logger.info(f"Cleaned batch {i//self.batch_size + 1}/{(len(data) + self.batch_size - 1)//self.batch_size}")
                
                # Save intermediate cleaned data
                if save_intermediate:
                    cleaned_file = file_path.replace('.json', '_cleaned.json').replace('.csv', '_cleaned.json')
                    await self._save_data_async(cleaned_data, cleaned_file)
                    logger.info(f"Saved cleaned data to: {cleaned_file}")
                
                data = cleaned_data
            
            # Create documents
            logger.info("Creating documents from cleaned data...")
            all_documents = []
            
            # Process in batches to manage memory
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                batch_documents = await self.create_documents_batch(batch)
                all_documents.extend(batch_documents)
                
                logger.info(f"Processed document batch {i//self.batch_size + 1}/{(len(data) + self.batch_size - 1)//self.batch_size}")
            
            # Update stats
            self.processing_stats['total_documents'] = len(all_documents)
            self.processing_stats['processing_time'] = time.time() - start_time
            
            logger.info(f"Processing completed: {len(all_documents)} documents created in {self.processing_stats['processing_time']:.2f}s")
            
            return all_documents
            
        except Exception as e:
            logger.error(f"Error in async data processing: {e}")
            raise
    
    async def _save_data_async(self, data: List[Dict], file_path: str):
        """
        Save data to file asynchronously
        
        Args:
            data: Data to save
            file_path: Output file path
        """
        loop = asyncio.get_event_loop()
        
        def _save_file():
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(executor, _save_file)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats['total_items'] > 0:
            stats['success_rate'] = round((stats['processed_items'] / stats['total_items']) * 100, 2)
        else:
            stats['success_rate'] = 0
        
        if stats['processing_time'] > 0:
            stats['items_per_second'] = round(stats['processed_items'] / stats['processing_time'], 2)
            stats['documents_per_second'] = round(stats['total_documents'] / stats['processing_time'], 2)
        else:
            stats['items_per_second'] = 0
            stats['documents_per_second'] = 0
        
        return stats
    
    async def process_multiple_files(self, file_paths: List[str]) -> List[Document]:
        """
        Process multiple data files concurrently
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Combined list of documents from all files
        """
        logger.info(f"Processing {len(file_paths)} files concurrently...")
        
        # Create tasks for each file
        tasks = [
            self.process_data_async(file_path, clean_data=True, save_intermediate=True)
            for file_path in file_paths
        ]
        
        # Wait for all files to be processed
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing file {file_paths[i]}: {result}")
            elif isinstance(result, list):
                all_documents.extend(result)
                logger.info(f"File {file_paths[i]} contributed {len(result)} documents")
        
        logger.info(f"Total documents from all files: {len(all_documents)}")
        return all_documents


# Convenience functions
async def process_data_file_async(file_path: str, 
                                clean_data: bool = True,
                                chunk_size: int = 500,
                                chunk_overlap: int = 50) -> List[Document]:
    """
    Convenience function to process a single data file asynchronously
    
    Args:
        file_path: Path to data file
        clean_data: Whether to clean data
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of processed documents
    """
    processor = AsyncDataProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    documents = await processor.process_data_async(file_path, clean_data=clean_data)
    
    # Log final stats
    stats = processor.get_processing_stats()
    logger.info(f"Final processing stats: {stats}")
    
    return documents


if __name__ == "__main__":
    # Test async data processing
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        # Test with existing data file
        data_file = "data/output.json"
        if os.path.exists(data_file):
            documents = await process_data_file_async(data_file)
            print(f"Processed {len(documents)} documents")
        else:
            print(f"Data file not found: {data_file}")
    
    asyncio.run(main())
