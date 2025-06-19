"""
Enhanced Optimized Crawler with Advanced Caching System
Complete crawler with smart filtering, duplicate detection, and comprehensive caching
"""

import asyncio
import json
import csv
import os
import re
import hashlib
import sqlite3
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, urlunparse
from typing import Set, List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from pathlib import Path
import time
from collections import defaultdict, Counter
import urllib.robotparser

# Required imports
try:
    import requests
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright
    import PyPDF2
    import pdfplumber
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from difflib import SequenceMatcher
except ImportError as e:
    print(f"Error: Missing required library. Please install: {e}")
    print("Run: pip install requests beautifulsoup4 playwright PyPDF2 pdfplumber")
    print("Then run: playwright install")
    exit(1)


@dataclass
class CrawlConfig:
    """Enhanced configuration class for crawler settings"""
    # Basic crawling settings
    max_depth: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50
    request_delay: float = 1.0
    baak_delay: float = 2.0
    pdf_delay: float = 2.0
    similarity_threshold: float = 0.8
    duplicate_threshold: float = 0.95
    max_retries: int = 3
    timeout: int = 60
    max_concurrent: int = 5
    
    # Advanced caching settings
    enable_url_cache: bool = True
    enable_content_cache: bool = True
    enable_response_cache: bool = True
    enable_smart_filtering: bool = True
    enable_robots_respect: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 1000
    
    # Database settings
    enable_incremental_updates: bool = True
    database_path: str = "cache/crawler_cache.db"
    
    # Content filtering
    min_content_length: int = 100
    max_url_length: int = 500


@dataclass
class PageData:
    """Enhanced data structure for page information"""
    url: str
    title: str
    text_content: str
    source_type: str
    timestamp: str
    content_hash: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'url': self.url,
            'title': self.title,
            'text_content': self.text_content,
            'source_type': self.source_type,
            'timestamp': self.timestamp,
            'content_hash': self.content_hash,
            'metadata': self.metadata
        }


class AdvancedCacheManager:
    """Advanced caching system with multiple cache types and persistence"""
    
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Multiple cache types
        self.url_cache: Dict[str, Dict] = {}
        self.content_cache: Dict[str, PageData] = {}
        self.response_cache: Dict[str, Dict] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.robots_cache: Dict[str, Dict] = {}
        self.redirect_cache: Dict[str, str] = {}
        
        # Statistics
        self.cache_stats = defaultdict(int)
        
        # Initialize persistent storage
        self._init_database()
        self._load_cache()
    
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
            # Load from JSON files for quick access
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
                            # Convert string keys back to tuples
                            self.similarity_cache = {eval(k): v for k, v in data.items()}
                        else:
                            cache_dict.update(data)
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}")
    
    def save_cache(self):
        """Save cache to persistent storage"""
        try:
            # Save to JSON files
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
            
            # Save to database
            self._save_to_database()
            logging.info("Cache data saved to persistent storage")
            
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")
    
    def _save_to_database(self):
        """Save cache data to SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            # Save URL cache
            for url, data in self.url_cache.items():
                conn.execute("""
                    INSERT OR REPLACE INTO url_cache 
                    (url, content_hash, last_crawled, status, data)
                    VALUES (?, ?, ?, ?, ?)
                """, (url, data.get('content_hash'), data.get('cached_at'), 
                     data.get('status'), json.dumps(data)))
            
            # Save similarity cache
            for (hash1, hash2), similarity in self.similarity_cache.items():
                conn.execute("""
                    INSERT OR REPLACE INTO content_similarity
                    (hash1, hash2, similarity, calculated_at)
                    VALUES (?, ?, ?, ?)
                """, (hash1, hash2, similarity, datetime.now().isoformat()))
    
    def is_url_cached(self, url: str) -> Tuple[bool, Optional[Dict]]:
        """Check if URL is cached and return cached data"""
        if not self.config.enable_url_cache:
            return False, None
        
        if url in self.url_cache:
            cached_data = self.url_cache[url]
            cached_time = datetime.fromisoformat(cached_data.get('cached_at', '1970-01-01'))
            
            # Check if cache is still valid
            if datetime.now() - cached_time < timedelta(seconds=self.config.cache_ttl):
                self.cache_stats['url_hits'] += 1
                return True, cached_data
            else:
                # Cache expired
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
            # Remove oldest entries
            sorted_items = sorted(self.url_cache.items(), 
                                key=lambda x: x[1].get('cached_at', ''))
            for old_url, _ in sorted_items[:len(self.url_cache) - self.config.max_cache_size]:
                del self.url_cache[old_url]
    
    def get_content_similarity(self, hash1: str, hash2: str) -> Optional[float]:
        """Get cached content similarity"""
        if not self.config.enable_content_cache:
            return None
        
        key = (hash1, hash2) if hash1 < hash2 else (hash2, hash1)
        if key in self.similarity_cache:
            self.cache_stats['similarity_hits'] += 1
            return self.similarity_cache[key]
        
        self.cache_stats['similarity_misses'] += 1
        return None
    
    def cache_content_similarity(self, hash1: str, hash2: str, similarity: float):
        """Cache content similarity calculation"""
        if not self.config.enable_content_cache:
            return
        
        key = (hash1, hash2) if hash1 < hash2 else (hash2, hash1)
        self.similarity_cache[key] = similarity
    
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
        
        logging.info(f"Cache cleanup completed. Removed {len(expired_urls)} expired entries")
    
    def get_cache_statistics(self) -> Dict:
        """Get comprehensive cache statistics"""
        total_requests = sum(self.cache_stats.values())
        hit_rate = 0
        if total_requests > 0:
            hits = self.cache_stats['url_hits'] + self.cache_stats['similarity_hits']
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


class SmartUrlFilter:
    """Smart URL filtering to prevent duplicate crawling"""
    
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.crawled_patterns: Set[str] = set()
        self.url_fingerprints: Set[str] = set()
        self.domain_stats = Counter()
        
        # Common duplicate patterns
        self.duplicate_patterns = [
            r'/page/\d+',           # Pagination
            r'\?page=\d+',          # Query pagination
            r'\?p=\d+',             # Short pagination
            r'\?offset=\d+',        # Offset pagination
            r'\?start=\d+',         # Start pagination
            r'\?utm_',              # UTM parameters
            r'\?fbclid=',           # Facebook click IDs
            r'\?gclid=',            # Google click IDs
            r'\?ref=',              # Referrer parameters
            r'\?source=',           # Source parameters
            r'/tag/',               # Tag pages
            r'/category/',          # Category pages
            r'/author/',            # Author pages
            r'/date/',              # Date archives
            r'/\d{4}/\d{2}/',       # Date paths
            r'#.*$',                # Fragments
            r'\?print=',            # Print versions
            r'\?share=',            # Share parameters
            r'/feed/',              # RSS feeds
            r'/wp-admin/',          # WordPress admin
            r'/wp-content/',        # WordPress content
            r'/wp-includes/',       # WordPress includes
        ]
    
    def should_crawl_url(self, url: str, visited_urls: Set[str]) -> Tuple[bool, str]:
        """Determine if URL should be crawled"""
        if not self.config.enable_smart_filtering:
            return True, "filtering_disabled"
        
        # Basic checks
        if url in visited_urls:
            return False, "already_visited"
        
        if len(url) > self.config.max_url_length:
            return False, "url_too_long"
        
        # Check for duplicate patterns
        for pattern in self.duplicate_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False, f"matches_pattern: {pattern}"
        
        # Generate URL fingerprint
        fingerprint = self._generate_url_fingerprint(url)
        if fingerprint in self.url_fingerprints:
            return False, f"similar_url_pattern (fingerprint: {fingerprint[:8]})"
        
        # Domain limiting
        domain = urlparse(url).netloc
        if self.domain_stats[domain] > 100:  # Limit per domain
            return False, f"domain_limit_exceeded: {domain}"
        
        return True, "passed_all_filters"
    
    def add_crawled_url(self, url: str):
        """Add URL to crawled tracking"""
        fingerprint = self._generate_url_fingerprint(url)
        self.url_fingerprints.add(fingerprint)
        
        domain = urlparse(url).netloc
        self.domain_stats[domain] += 1
    
    def _generate_url_fingerprint(self, url: str) -> str:
        """Generate URL fingerprint for similarity detection"""
        parsed = urlparse(url)
        
        # Normalize path - remove numbers and IDs
        path = re.sub(r'/\d+', '/[ID]', parsed.path)
        path = re.sub(r'/[a-f0-9]{8,}', '/[HASH]', path)
        
        # Remove common parameters
        query_parts = []
        if parsed.query:
            for param in parsed.query.split('&'):
                key = param.split('=')[0]
                if not any(skip in key for skip in ['utm_', 'fb', 'gclid', 'ref', 'source']):
                    query_parts.append(key)
        
        fingerprint_data = f"{parsed.netloc}{path}{'?' + '&'.join(sorted(query_parts)) if query_parts else ''}"
        return hashlib.md5(fingerprint_data.encode()).hexdigest()


class RobotsChecker:
    """Robots.txt compliance checker with caching"""
    
    def __init__(self, cache_manager: AdvancedCacheManager):
        self.cache_manager = cache_manager
    
    async def get_robots_rules(self, domain: str) -> Dict:
        """Get robots.txt rules for domain with caching"""
        if domain in self.cache_manager.robots_cache:
            cached_data = self.cache_manager.robots_cache[domain]
            cached_time = datetime.fromisoformat(cached_data.get('cached_at', '1970-01-01'))
            
            # Cache robots.txt for 24 hours
            if datetime.now() - cached_time < timedelta(hours=24):
                return cached_data.get('rules', {})
        
        # Fetch robots.txt
        try:
            robots_url = f"https://{domain}/robots.txt"
            response = requests.get(robots_url, timeout=10)
            
            if response.status_code == 200:
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(robots_url)
                rp.read()
                
                rules = {
                    'disallow': [],
                    'allow': [],
                    'crawl_delay': None
                }
                
                # Parse robots.txt content
                for line in response.text.split('\n'):
                    line = line.strip()
                    if line.startswith('Disallow:'):
                        rules['disallow'].append(line.split(':', 1)[1].strip())
                    elif line.startswith('Allow:'):
                        rules['allow'].append(line.split(':', 1)[1].strip())
                    elif line.startswith('Crawl-delay:'):
                        try:
                            rules['crawl_delay'] = float(line.split(':', 1)[1].strip())
                        except ValueError:
                            pass
                
                # Cache the rules
                self.cache_manager.robots_cache[domain] = {
                    'rules': rules,
                    'cached_at': datetime.now().isoformat()
                }
                
                return rules
            
        except Exception as e:
            logging.warning(f"Failed to fetch robots.txt for {domain}: {e}")
        
        # Default to allow all
        return {'disallow': [], 'allow': [''], 'crawl_delay': None}
    
    def is_url_allowed(self, url: str, rules: Dict) -> bool:
        """Check if URL is allowed by robots.txt rules"""
        parsed = urlparse(url)
        path = parsed.path
        
        # Check disallow rules
        for pattern in rules.get('disallow', []):
            if pattern and path.startswith(pattern):
                return False
        
        # Check allow rules
        for pattern in rules.get('allow', []):
            if pattern and path.startswith(pattern):
                return True
        
        return True


class EnhancedContentManager:
    """Enhanced content management with incremental updates and similarity detection"""
    def __init__(self, data_dir: str = "data", config: CrawlConfig = None):
        self.config = config or CrawlConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
        # File paths
        self.json_path = self.data_dir / "output.json"
        self.csv_path = self.data_dir / "output.csv"
        self.index_path = self.data_dir / "content_index.json"
        
        # Initialize cache manager
        self.cache_manager = AdvancedCacheManager(self.config)
        
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
            cached_similarity = self.cache_manager.get_content_similarity(content_hash, existing_hash)
            
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
                    self.cache_manager.cache_content_similarity(content_hash, existing_hash, similarity)
                else:
                    similarity = 0.0
            
            if similarity >= self.config.duplicate_threshold:
                return True, f"similar_content (similarity: {similarity:.2f})", similarity
        
        return False, "unique_content", 0.0
    
    def should_update_content(self, url: str, new_content: str, config: CrawlConfig) -> Tuple[bool, str]:
        """Determine if content should be updated"""
        if not config.enable_incremental_updates:
            return True, "incremental_updates_disabled"
        
        if url not in self.existing_content:
            return True, "new_url"
        
        existing_item = self.existing_content[url]
        existing_hash = existing_item.get('content_hash')
        new_hash = self._calculate_content_hash(new_content)
        
        if existing_hash != new_hash:
            # Content changed
            existing_timestamp = datetime.fromisoformat(existing_item.get('timestamp', '1970-01-01'))
            if datetime.now() - existing_timestamp > timedelta(hours=24):
                return True, "content_changed_after_24h"
            else:
                return True, "content_changed"
        
        return False, "content_unchanged"
    
    def add_content_hash(self, content_hash: str, content: str):
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
        
        # Convert PageData to dictionaries
        new_data = [page.to_dict() for page in page_data]
        
        # Handle incremental updates
        if incremental and self.json_path.exists():
            try:
                # Load existing data
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Merge new data with existing (avoiding duplicates)
                existing_urls = {item['url'] for item in existing_data}
                
                merged_data = existing_data.copy()
                new_count = 0
                updated_count = 0
                
                for new_item in new_data:
                    if new_item['url'] in existing_urls:
                        # Update existing item
                        for i, existing_item in enumerate(merged_data):
                            if existing_item['url'] == new_item['url']:
                                if existing_item.get('content_hash') != new_item.get('content_hash'):
                                    merged_data[i] = new_item
                                    updated_count += 1
                                break
                    else:
                        # Add new item
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


class OptimizedCrawler:
    """Main optimized crawler with advanced caching and smart filtering"""
    
    def __init__(self, target_urls: List[str], config: CrawlConfig = None):
        self.config = config or CrawlConfig()
        self.target_urls = target_urls
        
        # Initialize components
        self.content_manager = EnhancedContentManager(config=self.config)
        self.url_filter = SmartUrlFilter(self.config)
        self.robots_checker = RobotsChecker(self.content_manager.cache_manager)
        
        # URL management
        self.visited_urls: Set[str] = set()
        self.pdf_urls: Set[str] = set()
        
        # Setup logging
        self._setup_logging()
        
        # Statistics
        self.stats = {
            'pages_crawled': 0,
            'pages_updated': 0,
            'pages_skipped': 0,
            'pdfs_processed': 0,
            'errors': 0,
            'cache_hits': 0,
            'duplicates_skipped': 0
        }
    
    def _setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
    
    async def crawl_page(self, url: str, depth: int = 0) -> Tuple[Optional[PageData], Set[str]]:
        """Crawl a single page with advanced caching and filtering"""
        if depth > self.config.max_depth:
            return None, set()
        
        # Smart URL filtering
        should_crawl, reason = self.url_filter.should_crawl_url(url, self.visited_urls)
        if not should_crawl:
            self.logger.debug(f"Skipping {url}: {reason}")
            return None, set()
        
        self.visited_urls.add(url)
        self.url_filter.add_crawled_url(url)
        
        # Check cache first
        is_cached, cached_data = self.content_manager.cache_manager.is_url_cached(url)
        if is_cached:
            self.stats['cache_hits'] += 1
            self.logger.info(f"Cache hit for {url}")
            
            # Create PageData from cached data
            if cached_data.get('status') == 'success':
                page_data = PageData(
                    url=url,
                    title=cached_data.get('title', ''),
                    text_content=cached_data.get('content', ''),
                    source_type='html',
                    timestamp=cached_data.get('cached_at', datetime.now().isoformat()),
                    content_hash=cached_data.get('content_hash', ''),
                    metadata={'cache_hit': True}
                )
                return page_data, set()
        
        # Robots.txt check
        if self.config.enable_robots_respect:
            domain = urlparse(url).netloc
            robots_rules = await self.robots_checker.get_robots_rules(domain)
            
            if not self.robots_checker.is_url_allowed(url, robots_rules):
                self.logger.info(f"Blocked by robots.txt: {url}")
                return None, set()
            
            # Respect crawl delay
            crawl_delay = robots_rules.get('crawl_delay')
            if crawl_delay:
                await asyncio.sleep(crawl_delay)
        
        # Request delay
        delay = self.config.baak_delay if 'baak' in url else self.config.request_delay
        await asyncio.sleep(delay)
        
        try:
            # Fetch content
            session = requests.Session()
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            response = session.get(
                url,
                timeout=self.config.timeout,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content
            title = soup.find('title').get_text().strip() if soup.find('title') else ''
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            text_content = soup.get_text()
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            # Content length check
            if len(text_content) < self.config.min_content_length:
                self.logger.debug(f"Content too short for {url}")
                return None, set()
            
            # Extract links
            links = set()
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(url, link['href'])
                if self._is_valid_url(absolute_url):
                    links.add(absolute_url)
            
            # Check for duplicate content
            if self.config.enable_content_cache:
                is_duplicate, duplicate_reason, similarity = self.content_manager.is_duplicate_content(text_content)
                if is_duplicate:
                    self.logger.info(f"Skipping duplicate content for {url}: {duplicate_reason}")
                    self.stats['duplicates_skipped'] += 1
                    return None, links
            
            # Check if content should be updated
            content_hash = self.content_manager._calculate_content_hash(text_content)
            should_update, reason = self.content_manager.should_update_content(url, text_content, self.config)
            
            if not should_update:
                self.logger.info(f"Skipping {url}: {reason}")
                self.stats['pages_skipped'] += 1
                return None, links
            
            # Add content hash to tracking
            self.content_manager.add_content_hash(content_hash, text_content)
            
            # Cache the crawl result
            self.content_manager.cache_crawl_result(url, text_content, title, "success")
            
            # Create page data
            page_data = PageData(
                url=url,
                title=title,
                text_content=text_content,
                source_type="html",
                timestamp=datetime.now().isoformat(),
                content_hash=content_hash,
                metadata={
                    'depth': depth,
                    'content_length': len(text_content),
                    'domain': urlparse(url).netloc
                }
            )
            
            self.stats['pages_crawled'] += 1
            if reason.startswith('content_changed'):
                self.stats['pages_updated'] += 1
            
            return page_data, links
            
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {e}")
            self.stats['errors'] += 1
            return None, set()
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for crawling"""
        parsed = urlparse(url)
        return (
            parsed.scheme in ['http', 'https'] and
            parsed.netloc and
            not any(ext in parsed.path.lower() for ext in ['.jpg', '.png', '.gif', '.css', '.js', '.ico'])
        )
    
    async def crawl_all_pages(self) -> List[PageData]:
        """Crawl all pages with BFS approach"""
        self.logger.info("Starting optimized crawling...")
        
        crawled_data = []
        current_urls = set(self.target_urls)
        
        for depth in range(self.config.max_depth + 1):
            if not current_urls:
                break
                
            self.logger.info(f"Crawling depth {depth}: {len(current_urls)} URLs")
            next_urls = set()
            
            # Process URLs at current depth
            for url in current_urls:
                if url in self.visited_urls:
                    continue
                    
                try:
                    page_data, links = await self.crawl_page(url, depth)
                    
                    if page_data:
                        crawled_data.append(page_data)
                    
                    # Add new links for next depth
                    if depth < self.config.max_depth:
                        for link in links:
                            if link not in self.visited_urls:
                                next_urls.add(link)
                                
                except Exception as e:
                    self.logger.error(f"Error crawling {url}: {e}")
                    self.stats['errors'] += 1
            
            current_urls = next_urls
        
        return crawled_data
    
    async def crawl(self, incremental: bool = True) -> Dict:
        """Main crawling method with comprehensive reporting"""
        start_time = time.time()
        
        try:
            # Cleanup expired cache
            self.content_manager.cache_manager.cleanup_expired_cache()
            
            # Crawl pages
            page_data = await self.crawl_all_pages()
            
            # Save data
            total_saved = self.content_manager.save_data(page_data, incremental=incremental)
            
            # Save cache
            self.content_manager.cache_manager.save_cache()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Get cache statistics
            cache_stats = self.content_manager.cache_manager.get_cache_statistics()
            
            # Generate report
            report = {
                'status': 'success',
                'crawl_summary': {
                    'total_pages_crawled': self.stats['pages_crawled'],
                    'pages_updated': self.stats['pages_updated'],
                    'pages_skipped': self.stats['pages_skipped'],
                    'duplicates_skipped': self.stats['duplicates_skipped'],
                    'cache_hits': self.stats['cache_hits'],
                    'errors': self.stats['errors'],
                    'total_saved': total_saved,
                    'duration_seconds': duration,
                    'urls_visited': len(self.visited_urls)
                },
                'cache_statistics': cache_stats,
                'configuration': {
                    'incremental_mode': incremental,
                    'similarity_threshold': self.config.similarity_threshold,
                    'duplicate_threshold': self.config.duplicate_threshold,
                    'max_depth': self.config.max_depth,
                    'caching_enabled': {
                        'url_cache': self.config.enable_url_cache,
                        'content_cache': self.config.enable_content_cache,
                        'smart_filtering': self.config.enable_smart_filtering
                    }
                }
            }
            
            self._print_report(report)
            return report
            
        except Exception as e:
            self.logger.error(f"Crawling failed: {e}")
            return {'status': 'error', 'error': str(e), 'stats': self.stats}
    
    def _print_report(self, report: Dict):
        """Print comprehensive crawling report"""
        print("\n" + "=" * 60)
        print("ENHANCED CRAWLER REPORT")
        print("=" * 60)
        
        summary = report['crawl_summary']
        cache_stats = report['cache_statistics']
        
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"URLs visited: {summary['urls_visited']}")
        print(f"Pages crawled: {summary['total_pages_crawled']}")
        print(f"Pages updated: {summary['pages_updated']}")
        print(f"Pages skipped: {summary['pages_skipped']}")
        print(f"Duplicates skipped: {summary['duplicates_skipped']}")
        print(f"Cache hits: {summary['cache_hits']}")
        print(f"Errors: {summary['errors']}")
        print(f"Total saved: {summary['total_saved']}")
        
        print(f"\nCache Statistics:")
        print(f"  Hit rate: {cache_stats['hit_rate']:.1f}%")
        print(f"  Cache sizes: {cache_stats['cache_sizes']}")
        
        print("=" * 60)


# Legacy function for backward compatibility
async def crawl_pipeline():
    """Legacy crawl pipeline function"""
    config = CrawlConfig(
        max_depth=2,
        enable_url_cache=True,
        enable_content_cache=True,
        enable_smart_filtering=True
    )
    
    target_urls = [
        "https://baak.gunadarma.ac.id/",
        "https://www.gunadarma.ac.id/"
    ]
    
    crawler = OptimizedCrawler(target_urls, config)
    await crawler.crawl(incremental=True)


if __name__ == "__main__":
    asyncio.run(crawl_pipeline())
