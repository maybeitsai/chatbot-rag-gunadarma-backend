"""
Advanced caching system with multiple cache types and persistence
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import defaultdict
import logging

from .config import CrawlConfig


class CacheManager:
    """Advanced caching system with multiple cache types and persistence"""
    
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Multiple cache types
        self.url_cache: Dict[str, Dict] = {}
        self.content_cache: Dict[str, str] = {}
        self.response_cache: Dict[str, Dict] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.robots_cache: Dict[str, Dict] = {}
        
        # Statistics
        self.cache_stats = defaultdict(int)
        
        # Initialize persistent storage
        self._init_database()
        self._load_cache()
        
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize SQLite database for persistent caching"""
        self.db_path = Path(self.config.database_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS url_cache (
                    url TEXT PRIMARY KEY,
                    content_hash TEXT,
                    last_crawled TIMESTAMP,
                    status TEXT,
                    data TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS content_similarity (
                    hash1 TEXT,
                    hash2 TEXT,
                    similarity REAL,
                    calculated_at TIMESTAMP,
                    PRIMARY KEY (hash1, hash2)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS robots_cache (
                    domain TEXT PRIMARY KEY,
                    rules TEXT,
                    cached_at TIMESTAMP
                )
            """)
    
    def _load_cache(self):
        """Load cache from persistent storage"""
        try:
            cache_files = {
                'url_cache.json': self.url_cache,
                'response_cache.json': self.response_cache,
                'similarity_cache.json': {},
                'robots_cache.json': self.robots_cache
            }
            
            for filename, cache_dict in cache_files.items():
                cache_file = self.cache_dir / filename
                if cache_file.exists():
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if filename == 'similarity_cache.json':
                            self.similarity_cache = {eval(k): v for k, v in data.items()}
                        else:
                            cache_dict.update(data)
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
    
    def save_cache(self):
        """Save cache to persistent storage"""
        try:
            cache_data = {
                'url_cache.json': self.url_cache,
                'response_cache.json': self.response_cache,
                'similarity_cache.json': {str(k): v for k, v in self.similarity_cache.items()},
                'robots_cache.json': self.robots_cache
            }
            
            for filename, data in cache_data.items():
                cache_file = self.cache_dir / filename
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self._save_to_database()
            self.logger.info("Cache data saved to persistent storage")
            
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    def _save_to_database(self):
        """Save cache data to SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            for url, data in self.url_cache.items():
                conn.execute("""
                    INSERT OR REPLACE INTO url_cache 
                    (url, content_hash, last_crawled, status, data)
                    VALUES (?, ?, ?, ?, ?)
                """, (url, data.get('content_hash'), data.get('cached_at'), 
                     data.get('status'), json.dumps(data)))
    
    def is_url_cached(self, url: str) -> Tuple[bool, Optional[Dict]]:
        """Check if URL is cached and return cached data"""
        if not self.config.enable_url_cache:
            return False, None
        
        if url in self.url_cache:
            cached_data = self.url_cache[url]
            cached_time = datetime.fromisoformat(cached_data.get('cached_at', '1970-01-01'))
            
            if datetime.now() - cached_time < timedelta(seconds=self.config.cache_ttl):
                self.cache_stats['url_hits'] += 1
                return True, cached_data
            else:
                del self.url_cache[url]
        
        self.cache_stats['url_misses'] += 1
        return False, None
    
    def cache_url_result(self, url: str, result: Dict):
        """Cache URL crawling result"""
        if not self.config.enable_url_cache:
            return
        
        self.url_cache[url] = {
            **result,
            'cached_at': datetime.now().isoformat()
        }
        
        # Limit cache size
        if len(self.url_cache) > self.config.max_cache_size:
            sorted_items = sorted(self.url_cache.items(), 
                                key=lambda x: x[1].get('cached_at', ''))
            for old_url, _ in sorted_items[:len(self.url_cache) - self.config.max_cache_size]:
                del self.url_cache[old_url]
    
    def get_cache_statistics(self) -> Dict:
        """Get comprehensive cache statistics"""
        total_requests = sum(self.cache_stats.values())
        hit_rate = 0
        if total_requests > 0:
            hits = self.cache_stats['url_hits']
            hit_rate = (hits / total_requests) * 100
        
        return {
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cache_sizes': {
                'url_cache': len(self.url_cache),
                'content_cache': len(self.content_cache),
                'response_cache': len(self.response_cache),
                'similarity_cache': len(self.similarity_cache),
                'robots_cache': len(self.robots_cache)
            },
            'cache_stats': dict(self.cache_stats)
        }
    
    def cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        current_time = datetime.now()
        expired_urls = []
        
        for url, data in self.url_cache.items():
            cached_time = datetime.fromisoformat(data.get('cached_at', '1970-01-01'))
            if current_time - cached_time > timedelta(seconds=self.config.cache_ttl):
                expired_urls.append(url)
        
        for url in expired_urls:
            del self.url_cache[url]
        
        self.logger.info(f"Cache cleanup completed. Removed {len(expired_urls)} expired entries")
    
    def cache_content_similarity(self, hash1: str, hash2: str, similarity: float):
        """Cache content similarity between two hashes"""
        key = (hash1, hash2) if hash1 < hash2 else (hash2, hash1)
        self.similarity_cache[key] = similarity
    
    def get_content_similarity(self, hash1: str, hash2: str) -> Optional[float]:
        """Get cached content similarity between two hashes"""
        key = (hash1, hash2) if hash1 < hash2 else (hash2, hash1)
        return self.similarity_cache.get(key)