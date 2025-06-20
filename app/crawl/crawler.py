"""
Main optimized crawler with advanced caching and smart filtering
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from urllib.parse import urlparse, urljoin
from typing import Set, List, Dict, Optional, Tuple
from collections import defaultdict
import re

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import CrawlConfig
from .models import PageData
from .content_manager import ContentManager
from .url_filter import UrlFilter
from .robots_checker import RobotsChecker


class WebCrawler:
    """Main optimized crawler with advanced caching and smart filtering"""
    
    def __init__(self, target_urls: List[str], config: CrawlConfig = None):
        self.config = config or CrawlConfig()
        self.target_urls = target_urls
        
        # Initialize components
        self.content_manager = ContentManager(config=self.config)
        self.url_filter = UrlFilter(self.config)
        self.robots_checker = RobotsChecker(self.config)
        
        # URL management
        self.visited_urls: Set[str] = set()
        self.pdf_urls: Set[str] = set()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'pages_crawled': 0,
            'pages_updated': 0,            'pages_skipped': 0,            'pdfs_processed': 0,
            'errors': 0,
            'cache_hits': 0,
            'duplicates_skipped': 0
        }
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and relevant for crawling"""
        try:
            parsed = urlparse(url)
            
            # Must have netloc and valid scheme
            if not parsed.netloc or parsed.scheme not in ('http', 'https'):
                return False
            
            # Skip file extensions that are not relevant
            path = parsed.path.lower()
            invalid_extensions = {
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',  # Images
                '.css', '.js',  # Stylesheets and scripts
                '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',  # Documents
                '.zip', '.rar', '.tar', '.gz',  # Archives
                '.mp4', '.mp3', '.avi', '.mov', '.wmv',  # Media
                '.xml', '.json'  # Data files (unless specifically needed)
            }
            
            for ext in invalid_extensions:
                if path.endswith(ext):
                    return False
            
            return True
        except:
            return False
    
    async def crawl_page(self, url: str, depth: int = 0) -> Tuple[Optional[PageData], Set[str]]:
        """Crawl a single page with advanced caching and filtering"""
        if depth > self.config.max_depth:
            return None, set()
        
        # Smart URL filtering
        should_crawl, reason = self.url_filter.should_crawl_url(url, self.visited_urls)
        if not should_crawl:
            self.logger.debug(f"Skipping {url}: {reason}")
            return None, set()
        
        # Check robots.txt
        can_crawl, robots_reason = self.robots_checker.can_crawl(url)
        if not can_crawl:
            self.logger.debug(f"Skipping {url}: {robots_reason}")
            return None, set()
        
        self.visited_urls.add(url)
        self.url_filter.add_crawled_url(url)
        
        # Check cache first
        is_cached, cached_data = self.content_manager.cache_manager.is_url_cached(url)
        if is_cached:
            self.stats['cache_hits'] += 1
            self.logger.info(f"Cache hit for {url}")
            
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
          # Request delay from robots.txt or config
        delay = self.robots_checker.get_crawl_delay(url)
        await asyncio.sleep(delay)
        
        try:
            # For BAAK sites, use special handling
            if "baak.gunadarma.ac.id" in url:
                return await self._crawl_baak_site(url, depth)
            
            # For other sites, use standard requests
            page_data, links = await self._crawl_standard_site(url, depth)
            return page_data, links
            
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {e}")
            self.stats['errors'] += 1
            return None, set()
    
    async def _crawl_standard_site(self, url: str, depth: int) -> Tuple[Optional[PageData], Set[str]]:
        """Crawl standard websites using requests"""
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
        links = self._extract_links_from_soup(soup, url)
        
        # Check for duplicate content
        if self.config.enable_content_cache:
            is_duplicate, duplicate_reason, similarity = self.content_manager.is_duplicate_content(text_content)
            if is_duplicate:
                self.logger.info(f"Skipping duplicate content for {url}: {duplicate_reason}")
                self.stats['duplicates_skipped'] += 1
                return None, links
        
        # Check if content should be updated
        content_hash = self.content_manager._calculate_content_hash(text_content)
        should_update, reason = self.content_manager.should_update_content(url, text_content)
        
        if not should_update:
            self.logger.info(f"Skipping {url}: {reason}")
            self.stats['pages_skipped'] += 1
            return None, links
        
        # Add content hash to tracking
        self.content_manager.add_content_hash(content_hash)
        
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
    
    async def _crawl_baak_site(self, url: str, depth: int) -> Tuple[Optional[PageData], Set[str]]:
        """Special handling for BAAK sites with Cloudflare protection"""
        self.logger.info("BAAK site detected - using Playwright with Cloudflare bypass")
        
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            self.logger.error("Playwright not installed. Install with: pip install playwright")
            return None, set()
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-blink-features=AutomationControlled',
                        '--disable-extensions',
                        '--disable-gpu',
                        '--no-first-run',
                        '--no-zygote'
                    ]
                )
                
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080}
                )
                
                page = await context.new_page()
                
                # Navigate to page
                response = await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                
                if not response or response.status != 200:
                    await browser.close()
                    return None, set()
                
                # Wait for content to load
                await page.wait_for_timeout(5000)
                await page.wait_for_load_state("networkidle", timeout=30000)
                
                # Get page content
                content = await page.content()
                title = await page.title()
                
                await browser.close()
                
                if len(content) < 1000 and ("cloudflare" in content.lower() or "access denied" in content.lower()):
                    self.logger.warning(f"Possible Cloudflare block detected for {url}")
                    return None, set()
                
                # Process content
                soup = BeautifulSoup(content, 'html.parser')
                links = self._extract_links_from_soup(soup, url)
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    element.decompose()
                
                text_content = soup.get_text()
                text_content = re.sub(r'\s+', ' ', text_content).strip()
                
                if len(text_content) < self.config.min_content_length:
                    return None, links
                
                # Create page data
                content_hash = hashlib.md5(text_content.encode()).hexdigest()
                page_data = PageData(
                    url=url,
                    title=title,
                    text_content=text_content,
                    source_type='html',
                    timestamp=datetime.now().isoformat(),
                    content_hash=content_hash,
                    metadata={'method': 'playwright_baak', 'depth': depth}
                )
                
                self.stats['pages_crawled'] += 1
                return page_data, links
                
        except Exception as e:
            self.logger.error(f"Playwright crawl error for BAAK URL {url}: {e}")
            return None, set()
    
    def _extract_links_from_soup(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        """Extract and normalize links from BeautifulSoup object"""
        links = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            if not href:
                continue
                
            absolute_url = urljoin(base_url, href)
            parsed = urlparse(absolute_url)
            
            if parsed.scheme in ('http', 'https') and parsed.netloc:
                if absolute_url.lower().endswith('.pdf'):
                    self.pdf_urls.add(absolute_url)
                else:
                    links.add(absolute_url)
        
        return links
    
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
    
    crawler = WebCrawler(target_urls, config)
    await crawler.crawl(incremental=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(crawl_pipeline())
