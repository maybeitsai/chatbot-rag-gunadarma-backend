import os
import json
import pandas as pd
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class EnhancedDataProcessor:
    """Enhanced data processor that supports optimized crawler output"""
    
    def __init__(self):
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 500))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load data from JSON or CSV file with enhanced error handling"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                data = df.to_dict('records')
            else:
                raise ValueError("Unsupported file format. Use JSON or CSV.")
            
            print(f"✅ Successfully loaded {len(data)} items from {file_path}")
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Empty CSV file: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")
    
    def validate_item(self, item: Dict) -> bool:
        """Validate if an item has required fields and content"""
        required_fields = ['url', 'text_content']
        
        # Check required fields
        for field in required_fields:
            if field not in item or not item[field]:
                return False
        
        # Check content length
        if len(item['text_content'].strip()) < 10:
            return False
        
        return True
    
    def extract_metadata(self, item: Dict) -> Dict:
        """Extract and enhance metadata from crawler output"""
        metadata = {
            'url': item.get('url', ''),
            'title': item.get('title', ''),
            'source_type': item.get('source_type', 'unknown'),
            'timestamp': item.get('timestamp', ''),
            'content_hash': item.get('content_hash', ''),
            'content_length': len(item.get('text_content', '')),
        }
        
        # Add enhanced metadata from optimized crawler
        if 'metadata' in item and isinstance(item['metadata'], dict):
            crawler_metadata = item['metadata']
            metadata.update({
                'crawl_depth': crawler_metadata.get('depth', 0),
                'update_reason': crawler_metadata.get('update_reason', ''),
                'original_content_length': crawler_metadata.get('content_length', 0)
            })
        
        # Parse timestamp and add date info
        try:
            if metadata['timestamp']:
                parsed_time = datetime.fromisoformat(metadata['timestamp'].replace('Z', '+00:00'))
                metadata.update({
                    'crawl_date': parsed_time.strftime('%Y-%m-%d'),
                    'crawl_year': parsed_time.year,
                    'crawl_month': parsed_time.month
                })
        except:
            pass
        
        # Extract domain from URL
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(metadata['url'])
            metadata['domain'] = parsed_url.netloc
            metadata['path'] = parsed_url.path
        except:
            metadata['domain'] = 'unknown'
            metadata['path'] = ''
        
        return metadata
    
    def create_documents(self, data: List[Dict]) -> List[Document]:
        """Convert raw data to LangChain documents with enhanced metadata"""
        documents = []
        skipped_count = 0
        
        for item in data:
            # Validate item
            if not self.validate_item(item):
                skipped_count += 1
                continue
            
            # Extract enhanced metadata
            metadata = self.extract_metadata(item)
            
            # Create document
            doc = Document(
                page_content=item['text_content'],
                metadata=metadata
            )
            documents.append(doc)
        
        if skipped_count > 0:
            print(f"⚠️  Skipped {skipped_count} invalid items")
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with enhanced metadata"""
        chunked_docs = []
        
        for doc_idx, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc.page_content)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create enhanced metadata for chunk
                chunk_metadata = {
                    **doc.metadata,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'document_index': doc_idx,
                    'chunk_id': f"{doc.metadata.get('content_hash', doc_idx)}_{chunk_idx}",
                    'chunk_length': len(chunk)
                }
                
                chunked_doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                chunked_docs.append(chunked_doc)
        
        return chunked_docs
    
    def filter_documents(self, documents: List[Document], 
                        min_content_length: int = 50,
                        exclude_types: Optional[List[str]] = None,
                        include_domains: Optional[List[str]] = None) -> List[Document]:
        """Filter documents based on various criteria"""
        
        if exclude_types is None:
            exclude_types = []
        
        filtered_docs = []
        
        for doc in documents:
            # Filter by content length
            if len(doc.page_content.strip()) < min_content_length:
                continue
            
            # Filter by source type
            if doc.metadata.get('source_type') in exclude_types:
                continue
            
            # Filter by domain
            if include_domains and doc.metadata.get('domain') not in include_domains:
                continue
            
            filtered_docs.append(doc)
        
        removed_count = len(documents) - len(filtered_docs)
        if removed_count > 0:
            print(f"🔍 Filtered out {removed_count} documents based on criteria")
        
        return filtered_docs
    
    def get_statistics(self, documents: List[Document]) -> Dict:
        """Generate statistics about the processed documents"""
        if not documents:
            return {}
        
        stats = {
            'total_documents': len(documents),
            'total_content_length': sum(len(doc.page_content) for doc in documents),
            'avg_content_length': sum(len(doc.page_content) for doc in documents) / len(documents),
            'source_types': {},
            'domains': {},
            'crawl_dates': {},
            'chunk_distribution': {}
        }
        
        for doc in documents:
            # Source type distribution
            source_type = doc.metadata.get('source_type', 'unknown')
            stats['source_types'][source_type] = stats['source_types'].get(source_type, 0) + 1
            
            # Domain distribution
            domain = doc.metadata.get('domain', 'unknown')
            stats['domains'][domain] = stats['domains'].get(domain, 0) + 1
            
            # Crawl date distribution
            crawl_date = doc.metadata.get('crawl_date', 'unknown')
            stats['crawl_dates'][crawl_date] = stats['crawl_dates'].get(crawl_date, 0) + 1
            
            # Chunk distribution
            total_chunks = doc.metadata.get('total_chunks', 1)
            stats['chunk_distribution'][total_chunks] = stats['chunk_distribution'].get(total_chunks, 0) + 1
        
        return stats
    
    def print_statistics(self, stats: Dict):
        """Print formatted statistics"""
        print("\n📊 DOCUMENT STATISTICS")
        print("=" * 50)
        print(f"Total documents: {stats['total_documents']}")
        print(f"Total content length: {stats['total_content_length']:,} characters")
        print(f"Average content length: {stats['avg_content_length']:.0f} characters")
        
        print(f"\n📄 Source Types:")
        for source_type, count in sorted(stats['source_types'].items()):
            print(f"  {source_type}: {count}")
        
        print(f"\n🌐 Domains:")
        for domain, count in sorted(stats['domains'].items()):
            print(f"  {domain}: {count}")
        
        if stats['crawl_dates']:
            print(f"\n📅 Recent Crawl Dates:")
            recent_dates = sorted(stats['crawl_dates'].items(), reverse=True)[:5]
            for date, count in recent_dates:
                print(f"  {date}: {count}")
        
        print("=" * 50)
    
    def process_data(self, file_path: str, 
                    filter_config: Optional[Dict] = None) -> List[Document]:
        """Complete data processing pipeline with enhanced features"""
        
        print(f"🔄 Processing data from {file_path}...")
        
        # Load data
        raw_data = self.load_data(file_path)
        
        # Create documents
        print("📄 Creating documents...")
        documents = self.create_documents(raw_data)
        print(f"✅ Created {len(documents)} documents")
        
        # Apply filters if specified
        if filter_config:
            print("🔍 Applying filters...")
            documents = self.filter_documents(documents, **filter_config)
            print(f"✅ {len(documents)} documents after filtering")
        
        # Chunk documents
        print("✂️  Chunking documents...")
        chunked_documents = self.chunk_documents(documents)
        print(f"✅ Created {len(chunked_documents)} chunks")
        
        # Generate and print statistics
        stats = self.get_statistics(chunked_documents)
        self.print_statistics(stats)
        
        return chunked_documents


# Backward compatibility
class DataProcessor(EnhancedDataProcessor):
    """Legacy DataProcessor class for backward compatibility"""
    pass


if __name__ == "__main__":
    processor = EnhancedDataProcessor()
    
    # Try to process JSON first, then CSV
    for file_path in ["data/output.json", "data/output.csv"]:
        if os.path.exists(file_path):
            # Configure filtering (optional)
            filter_config = {
                'min_content_length': 100,  # Minimum content length
                'exclude_types': [],        # Exclude certain source types
                'include_domains': None     # Only include specific domains
            }
            
            documents = processor.process_data(file_path, filter_config)
            print(f"\n🎉 Successfully processed {len(documents)} document chunks")
            break
    else:
        print("❌ No data files found. Please ensure data/output.json or data/output.csv exists.")
        print("💡 Run the crawler first to generate data files.")