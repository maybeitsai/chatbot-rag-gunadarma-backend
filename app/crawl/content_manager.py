"""
Enhanced content management with incremental updates and similarity detection
"""

import json
import csv
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple
from difflib import SequenceMatcher
import logging

from .config import CrawlConfig
from .models import PageData
from .cache_manager import CacheManager


class ContentManager:
    """Enhanced content management with incremental updates and similarity detection"""
    
    def __init__(self, data_dir: str = "data", config: CrawlConfig = None):
        self.config = config or CrawlConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # File paths
        self.json_path = self.data_dir / "output.json"
        self.csv_path = self.data_dir / "output.csv"
        self.index_path = self.data_dir / "content_index.json"
        
        # Initialize cache manager
        self.cache_manager = CacheManager(self.config)
        
        # Content tracking
        self.existing_content: Dict[str, Dict] = {}
        self.content_hashes: Set[str] = set()
        
        # Load existing content
        self._load_existing_content()
    
    def _load_existing_content(self):
        """Load existing content for incremental updates"""
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    
                for item in existing_data:
                    url = item.get('url')
                    content_hash = item.get('content_hash')
                    if url and content_hash:
                        self.existing_content[url] = item
                        self.content_hashes.add(content_hash)
                        
                self.logger.info(f"Loaded {len(self.existing_content)} existing content items")
                
            except Exception as e:
                self.logger.warning(f"Failed to load existing content: {e}")
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate content hash for duplicate detection"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def is_duplicate_content(self, content: str) -> Tuple[bool, str, float]:
        """Check if content is duplicate using caching"""
        content_hash = self._calculate_content_hash(content)
        
        # Exact duplicate check
        if content_hash in self.content_hashes:
            return True, "exact_duplicate", 1.0
        
        # Similarity check with caching
        for existing_hash in self.content_hashes:
            cached_similarity = self.cache_manager.similarity_cache.get((content_hash, existing_hash))
            
            if cached_similarity is not None:
                similarity = cached_similarity
            else:
                # Calculate similarity
                existing_content = ""
                for item in self.existing_content.values():
                    if item.get('content_hash') == existing_hash:
                        existing_content = item.get('text_content', '')
                        break
                
                if existing_content:
                    similarity = SequenceMatcher(None, content, existing_content).ratio()
                    # Cache the result
                    key = (content_hash, existing_hash) if content_hash < existing_hash else (existing_hash, content_hash)
                    self.cache_manager.similarity_cache[key] = similarity
                else:
                    similarity = 0.0
            
            if similarity >= self.config.duplicate_threshold:
                return True, f"similar_content (similarity: {similarity:.2f})", similarity
        
        return False, "unique_content", 0.0
    
    def should_update_content(self, url: str, new_content: str) -> Tuple[bool, str]:
        """Determine if content should be updated"""
        if not self.config.enable_incremental_updates:
            return True, "incremental_updates_disabled"
        
        if url not in self.existing_content:
            return True, "new_url"
        
        existing_item = self.existing_content[url]
        existing_hash = existing_item.get('content_hash')
        new_hash = self._calculate_content_hash(new_content)
        
        if existing_hash != new_hash:
            existing_timestamp = datetime.fromisoformat(existing_item.get('timestamp', '1970-01-01'))
            if datetime.now() - existing_timestamp > timedelta(hours=24):
                return True, "content_changed_after_24h"
            else:
                return True, "content_changed"
        
        return False, "content_unchanged"    
    def add_content_hash(self, content_hash: str, content: str = None):
        """Add content hash to tracking"""
        self.content_hashes.add(content_hash)
    
    def cache_crawl_result(self, url: str, content: str, title: str, status: str):
        """Cache crawl result"""
        result = {
            'content': content,
            'title': title,
            'status': status,
            'content_hash': self._calculate_content_hash(content),
            'content_length': len(content)
        }
        self.cache_manager.cache_url_result(url, result)
    
    def save_data(self, page_data: List[PageData], incremental: bool = True) -> int:
        """Save data to JSON and CSV with incremental update support"""
        if not page_data:
            self.logger.info("No new data to save")
            return 0
        
        new_data = [page.to_dict() for page in page_data]
        
        if incremental and self.json_path.exists():
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                existing_urls = {item['url'] for item in existing_data}
                merged_data = existing_data.copy()
                new_count = 0
                updated_count = 0
                
                for new_item in new_data:
                    if new_item['url'] in existing_urls:
                        for i, existing_item in enumerate(merged_data):
                            if existing_item['url'] == new_item['url']:
                                if existing_item.get('content_hash') != new_item.get('content_hash'):
                                    merged_data[i] = new_item
                                    updated_count += 1
                                break
                    else:
                        merged_data.append(new_item)
                        new_count += 1
                
                final_data = merged_data
                self.logger.info(f"Incremental update: {new_count} new items, {updated_count} updated items")
                
            except Exception as e:
                self.logger.error(f"Failed to load existing data for incremental update: {e}")
                final_data = new_data
        else:
            final_data = new_data
        
        try:
            # Save to JSON
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)
            
            # Save to CSV
            if final_data:
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=final_data[0].keys())
                    writer.writeheader()
                    writer.writerows(final_data)
            
            self.logger.info(f"Saved {len(final_data)} total items to {self.json_path} and {self.csv_path}")
            return len(final_data)
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            return 0