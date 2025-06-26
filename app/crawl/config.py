"""
Configuration settings for the crawler
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CrawlConfig:
    """Configuration class for crawler settings"""
    # Basic crawling settings
    max_depth: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50
    request_delay: float = 1.0
    baak_delay: float = 2.0
    pdf_delay: float = 2.0
    similarity_threshold: float = 0.92
    duplicate_threshold: float = 0.96
    max_retries: int = 3
    timeout: int = 60
    max_concurrent: int = 5
    
    # Caching settings
    enable_url_cache: bool = True
    enable_content_cache: bool = True
    enable_response_cache: bool = True
    enable_smart_filtering: bool = True
    enable_robots_respect: bool = True
    cache_ttl: int = 60 * 60 * 24
    
    # Robots.txt bypass for specific domains
    robots_bypass_domains: List[str] = field(default_factory=lambda: [
        'baak.gunadarma.ac.id',
    ])
    
    # Database settings
    enable_incremental_updates: bool = True
    database_path: str = "cache/crawler_cache.db"
    
    # Content filtering
    min_content_length: int = 100
    max_url_length: int = 500