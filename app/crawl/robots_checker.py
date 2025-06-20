"""
Robots.txt checker for respecting website crawling policies
"""

import requests
import logging
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from app.crawl.config import CrawlConfig


class RobotsChecker:
    """Robots.txt checker for respecting website crawling policies"""
    
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.robots_cache: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)
        
        # User agent for robots.txt checking
        self.user_agent = "ChatbotRAG-Crawler/1.0"
    
    def can_crawl(self, url: str) -> Tuple[bool, str]:
        """Check if URL can be crawled according to robots.txt"""
        if not self.config.enable_robots_respect:
            return True, "robots_check_disabled"
        
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Check if domain is in bypass list
        if parsed_url.netloc in self.config.robots_bypass_domains:
            return True, f"domain_bypassed: {parsed_url.netloc}"
        
        # Get or fetch robots.txt
        robots_parser = self._get_robots_parser(domain)
        
        if robots_parser is None:
            # If robots.txt doesn't exist or can't be fetched, allow crawling
            return True, "no_robots_txt"
        
        # Check if URL can be crawled
        can_crawl = robots_parser.can_fetch(self.user_agent, url)
        
        if can_crawl:
            return True, "robots_allowed"
        else:
            return False, "robots_disallowed"
    
    def get_crawl_delay(self, url: str) -> float:
        """Get crawl delay from robots.txt"""
        if not self.config.enable_robots_respect:
            return self.config.request_delay
        
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Check if domain is in bypass list
        if parsed_url.netloc in self.config.robots_bypass_domains:
            return self.config.baak_delay if 'baak' in parsed_url.netloc else self.config.request_delay
        
        robots_parser = self._get_robots_parser(domain)
        
        if robots_parser is None:
            return self.config.request_delay
        
        # Get crawl delay from robots.txt
        crawl_delay = robots_parser.crawl_delay(self.user_agent)
        
        if crawl_delay is not None:
            # Use robots.txt delay but respect minimum delay
            return max(crawl_delay, self.config.request_delay)
        else:
            return self.config.request_delay
    
    def _get_robots_parser(self, domain: str) -> Optional[RobotFileParser]:
        """Get robots.txt parser for domain with caching"""
        # Check cache first
        if domain in self.robots_cache:
            cached_data = self.robots_cache[domain]
            cached_time = datetime.fromisoformat(cached_data.get('cached_at', '1970-01-01'))
            
            # Cache for 24 hours
            if datetime.now() - cached_time < timedelta(hours=24):
                return cached_data.get('parser')
        
        # Fetch robots.txt
        robots_url = urljoin(domain, '/robots.txt')
        
        try:
            response = requests.get(
                robots_url,
                timeout=10,
                headers={'User-Agent': self.user_agent}
            )
            
            if response.status_code == 200:
                # Parse robots.txt
                rp = RobotFileParser()
                rp.set_url(robots_url)
                rp.read()
                
                # Cache the parser
                self.robots_cache[domain] = {
                    'parser': rp,
                    'cached_at': datetime.now().isoformat(),
                    'status': 'success'
                }
                
                self.logger.info(f"Successfully loaded robots.txt for {domain}")
                return rp
                
            elif response.status_code == 404:
                # No robots.txt file
                self.robots_cache[domain] = {
                    'parser': None,
                    'cached_at': datetime.now().isoformat(),
                    'status': 'not_found'
                }
                
                self.logger.debug(f"No robots.txt found for {domain}")
                return None
                
            else:
                self.logger.warning(f"Failed to fetch robots.txt for {domain}: HTTP {response.status_code}")
                return None
                
        except requests.RequestException as e:
            self.logger.warning(f"Error fetching robots.txt for {domain}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing robots.txt for {domain}: {e}")
            return None
    
    def get_sitemap_urls(self, url: str) -> list:
        """Get sitemap URLs from robots.txt"""
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        robots_parser = self._get_robots_parser(domain)
        
        if robots_parser is None:
            return []
        
        # Get sitemap URLs
        try:
            sitemaps = []
            for line in robots_parser._opener.open(robots_parser.url()).readlines():
                line = line.decode('utf-8').strip()
                if line.lower().startswith('sitemap:'):
                    sitemap_url = line.split(':', 1)[1].strip()
                    sitemaps.append(sitemap_url)
            
            return sitemaps
            
        except Exception as e:
            self.logger.error(f"Error extracting sitemaps from robots.txt for {domain}: {e}")
            return []
    
    def clear_cache(self):
        """Clear robots.txt cache"""
        self.robots_cache.clear()
        self.logger.info("Robots.txt cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get robots.txt cache information"""
        return {
            'cached_domains': len(self.robots_cache),
            'domains': list(self.robots_cache.keys()),
            'cache_details': {
                domain: {
                    'cached_at': data.get('cached_at'),
                    'status': data.get('status'),
                    'has_parser': data.get('parser') is not None
                }
                for domain, data in self.robots_cache.items()
            }
        }
