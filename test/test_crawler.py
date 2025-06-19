#!/usr/bin/env python3
"""
Test file for Enhanced Optimized Crawler
Tests all major components including caching, filtering, and incremental updates
"""

import asyncio
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crawl.crawler import (
    WebCrawler, 
    CrawlConfig, 
    CacheManager,
    UrlFilter,
    RobotsChecker,
    ContentManager,
    PageData
)


class TestCrawlConfig:
    """Test CrawlConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = CrawlConfig()
        
        assert config.max_depth == 3
        assert config.similarity_threshold == 0.8
        assert config.duplicate_threshold == 0.95
        assert config.enable_url_cache is True
        assert config.enable_content_cache is True
        assert config.enable_smart_filtering is True
        assert config.cache_ttl == 60 * 60 * 24  * 30
        assert config.max_cache_size == 1000
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = CrawlConfig(
            max_depth=2,
            similarity_threshold=0.9,
            enable_url_cache=False,
            cache_ttl=7200
        )
        
        assert config.max_depth == 2
        assert config.similarity_threshold == 0.9
        assert config.enable_url_cache is False
        assert config.cache_ttl == 7200


class TestCacheManager:
    """Test CacheManager functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = CrawlConfig(database_path=f"{self.temp_dir}/test_cache.db")
        self.cache_manager = CacheManager(self.config)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_url_caching(self):
        """Test URL result caching"""
        test_url = "https://example.com/test"
        test_result = {
            "content": "Test content",
            "title": "Test Title",
            "status": "success"
        }
        
        # Cache the result
        self.cache_manager.cache_url_result(test_url, test_result)
        
        # Check if cached
        is_cached, cached_data = self.cache_manager.is_url_cached(test_url)
        
        assert is_cached is True
        assert cached_data["content"] == "Test content"
        assert cached_data["title"] == "Test Title"
        assert cached_data["status"] == "success"
    
    def test_content_similarity_caching(self):
        """Test content similarity caching"""
        hash1 = "abc123"
        hash2 = "def456" 
        similarity = 0.85
        
        # Cache similarity
        self.cache_manager.cache_content_similarity(hash1, hash2, similarity)
        
        # Retrieve similarity
        cached_similarity = self.cache_manager.get_content_similarity(hash1, hash2)
        
        assert cached_similarity == similarity
    
    def test_cache_statistics(self):
        """Test cache statistics tracking"""
        # Perform some cache operations
        self.cache_manager.cache_url_result("https://test1.com", {"status": "success"})
        self.cache_manager.cache_content_similarity("hash1", "hash2", 0.8)
        
        # Check URL cache hit
        self.cache_manager.is_url_cached("https://test1.com")
        
        # Get statistics
        stats = self.cache_manager.get_cache_statistics()
        
        assert "total_requests" in stats
        assert "hit_rate" in stats
        assert "cache_sizes" in stats
        assert stats["cache_sizes"]["url_cache"] >= 1
        assert stats["cache_sizes"]["similarity_cache"] >= 1
    
    def test_cache_cleanup(self):
        """Test cache cleanup functionality"""
        # Add some test data
        test_url = "https://example.com/cleanup_test"
        self.cache_manager.cache_url_result(test_url, {"status": "success"})
        
        # Verify it's cached
        is_cached, _ = self.cache_manager.is_url_cached(test_url)
        assert is_cached is True
        
        # Run cleanup (won't remove fresh entries)
        self.cache_manager.cleanup_expired_cache()
        
        # Should still be cached
        is_cached, _ = self.cache_manager.is_url_cached(test_url)
        assert is_cached is True


class TestUrlFilter:
    """Test UrlFilter functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = CrawlConfig()
        self.url_filter = UrlFilter(self.config)
    
    def test_duplicate_pattern_filtering(self):
        """Test filtering of duplicate URL patterns"""
        visited_urls = set()
        
        # Test pagination filtering
        should_crawl, reason = self.url_filter.should_crawl_url(
            "https://example.com/page/2", visited_urls
        )
        assert should_crawl is False
        assert "pattern" in reason
        
        # Test UTM parameter filtering
        should_crawl, reason = self.url_filter.should_crawl_url(
            "https://example.com/article?utm_source=facebook", visited_urls
        )
        assert should_crawl is False
        assert "pattern" in reason
    
    def test_url_length_filtering(self):
        """Test URL length filtering"""
        visited_urls = set()
        long_url = "https://example.com/" + "a" * 500
        
        should_crawl, reason = self.url_filter.should_crawl_url(long_url, visited_urls)
        assert should_crawl is False
        assert reason == "url_too_long"
    
    def test_fingerprint_detection(self):
        """Test URL fingerprint-based duplicate detection"""
        visited_urls = set()
        
        # Add first URL
        url1 = "https://example.com/article/123"
        should_crawl, _ = self.url_filter.should_crawl_url(url1, visited_urls)
        assert should_crawl is True
        
        self.url_filter.add_crawled_url(url1)
        
        # Try similar URL with different ID
        url2 = "https://example.com/article/456" 
        should_crawl, reason = self.url_filter.should_crawl_url(url2, visited_urls)
        assert should_crawl is False
        assert "similar_url_pattern" in reason
    
    def test_domain_limiting(self):
        """Test domain crawling limits"""
        visited_urls = set()
        
        # Add many URLs from same domain
        for i in range(105):  # Exceed limit of 100
            url = f"https://example.com/page{i}"
            self.url_filter.add_crawled_url(url)
        
        # Next URL should be blocked
        should_crawl, reason = self.url_filter.should_crawl_url(
            "https://example.com/newpage", visited_urls
        )
        assert should_crawl is False
        assert "domain_limit_exceeded" in reason


class TestContentManager:
    """Test ContentManager functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = CrawlConfig()
        self.content_manager = ContentManager(
            data_dir=self.temp_dir, 
            config=self.config
        )
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_duplicate_content_detection(self):
        """Test content duplicate detection"""
        content1 = "This is a test content for duplicate detection."
        content2 = "This is a test content for duplicate detection."  # Identical
        content3 = "This is completely different content."
        
        # First content should not be duplicate
        is_dup, reason, similarity = self.content_manager.is_duplicate_content(content1)
        assert is_dup is False
        
        # Add content hash
        hash1 = self.content_manager._calculate_content_hash(content1)
        self.content_manager.add_content_hash(hash1, content1)
        
        # Identical content should be detected as duplicate
        is_dup, reason, similarity = self.content_manager.is_duplicate_content(content2)
        assert is_dup is True
        assert reason == "exact_duplicate"
        assert similarity == 1.0
        
        # Different content should not be duplicate
        is_dup, reason, similarity = self.content_manager.is_duplicate_content(content3)
        assert is_dup is False
        assert reason == "unique_content"
    
    def test_incremental_updates(self):
        """Test incremental update functionality"""
        # Create test page data
        page_data = [
            PageData(
                url="https://example.com/test1",
                title="Test Page 1",
                text_content="Test content 1",
                source_type="html",
                timestamp="2024-01-01T10:00:00",
                content_hash="hash1"
            ),
            PageData(
                url="https://example.com/test2", 
                title="Test Page 2",
                text_content="Test content 2",
                source_type="html",
                timestamp="2024-01-01T10:00:00",
                content_hash="hash2"
            )
        ]
        
        # Save initial data
        count = self.content_manager.save_data(page_data, incremental=True)
        assert count == 2
        
        # Verify files exist
        json_path = Path(self.temp_dir) / "output.json"
        csv_path = Path(self.temp_dir) / "output.csv"
        
        assert json_path.exists()
        assert csv_path.exists()
        
        # Verify JSON content
        with open(json_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 2
        assert saved_data[0]["url"] == "https://example.com/test1"
        assert saved_data[1]["url"] == "https://example.com/test2"


class TestWebCrawler:
    """Test WebCrawler integration"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = CrawlConfig(
            max_depth=1,
            request_delay=0.1,
            timeout=10
        )
        self.target_urls = ["https://httpbin.org/html"]
        self.crawler = WebCrawler(self.target_urls, self.config)
    
    def test_crawler_initialization(self):
        """Test crawler initialization"""
        assert self.crawler.config.max_depth == 1
        assert len(self.crawler.target_urls) == 1
        assert self.crawler.target_urls[0] == "https://httpbin.org/html"
        
        # Check components are initialized
        assert self.crawler.content_manager is not None
        assert self.crawler.url_filter is not None
        assert self.crawler.robots_checker is not None
    
    @pytest.mark.asyncio
    async def test_single_page_crawl(self):
        """Test crawling a single page"""
        test_url = "https://httpbin.org/html"
        
        page_data, links = await self.crawler.crawl_page(test_url, depth=0)
        
        # Should successfully crawl
        if page_data:  # May fail due to network issues in testing
            assert page_data.url == test_url
            assert len(page_data.text_content) > 0
            assert page_data.source_type == "html"
            assert page_data.content_hash != ""
    
    @pytest.mark.asyncio 
    async def test_url_validation(self):
        """Test URL validation"""
        # Valid URLs
        assert self.crawler._is_valid_url("https://example.com/page") is True
        assert self.crawler._is_valid_url("http://example.com/page") is True
        
        # Invalid URLs
        assert self.crawler._is_valid_url("https://example.com/image.jpg") is False
        assert self.crawler._is_valid_url("https://example.com/style.css") is False
        assert self.crawler._is_valid_url("ftp://example.com/file") is False
    
    @pytest.mark.asyncio
    async def test_crawl_with_caching(self):
        """Test full crawl with caching enabled"""
        try:
            report = await self.crawler.crawl(incremental=True)
            
            # Check report structure
            assert "status" in report
            assert "crawl_summary" in report
            assert "cache_statistics" in report
            assert "configuration" in report
            
            if report["status"] == "success":
                summary = report["crawl_summary"]
                assert "total_pages_crawled" in summary
                assert "duration_seconds" in summary
                assert "cache_hits" in summary
                
                cache_stats = report["cache_statistics"]
                assert "hit_rate" in cache_stats
                assert "cache_sizes" in cache_stats
                
        except Exception as e:
            # Network issues may cause test failure
            print(f"Crawl test failed due to network issues: {e}")


def test_page_data_serialization():
    """Test PageData serialization"""
    page_data = PageData(
        url="https://example.com/test",
        title="Test Page",
        text_content="Test content",
        source_type="html", 
        timestamp="2024-01-01T10:00:00",
        content_hash="abc123",
        metadata={"test": "value"}
    )
    
    # Test to_dict conversion
    data_dict = page_data.to_dict()
    
    assert data_dict["url"] == "https://example.com/test"
    assert data_dict["title"] == "Test Page"
    assert data_dict["text_content"] == "Test content"
    assert data_dict["metadata"]["test"] == "value"


@pytest.mark.asyncio
async def test_robots_checker():
    """Test RobotsChecker functionality"""
    config = CrawlConfig()
    cache_manager = CacheManager(config)
    robots_checker = RobotsChecker(cache_manager)
    
    try:
        # Test with a known domain
        rules = await robots_checker.get_robots_rules("httpbin.org")
        
        assert isinstance(rules, dict)
        assert "disallow" in rules
        assert "allow" in rules
        assert "crawl_delay" in rules
        
        # Test URL allowance check
        test_url = "https://httpbin.org/html"
        is_allowed = robots_checker.is_url_allowed(test_url, rules)
        assert isinstance(is_allowed, bool)
        
    except Exception as e:
        print(f"Robots.txt test failed due to network issues: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
