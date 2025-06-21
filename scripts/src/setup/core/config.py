# File: scripts/src/setup/core/config.py

from dataclasses import dataclass

@dataclass
class SetupConfig:
    """Configuration for setup process"""
    skip_crawling: bool = False
    force_crawl: bool = False
    cache_only: bool = False
    optimize_only: bool = False
    crawl_only: bool = False
    log_level: str = "INFO"