"""
Smart URL filtering to prevent duplicate crawling
"""

import re
import hashlib
from urllib.parse import urlparse
from typing import Set, Tuple
from collections import Counter

from .config import CrawlConfig


class UrlFilter:
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
        if self.domain_stats[domain] > 100:
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