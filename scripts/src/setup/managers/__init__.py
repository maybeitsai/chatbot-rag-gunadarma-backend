"""
Managers package for RAG System Setup.

This package contains management components responsible for different
aspects of the RAG system setup process, including caching, data crawling,
data processing, and system testing.

Modules:
    - cache_manager: Cache management and optimization
    - data_crawler: Web crawling and data collection
    - data_processor: Data processing and transformation
    - system_tester: System testing and validation

Each manager handles a specific domain of the setup process and can be
used independently or as part of the complete setup orchestration.
"""

try:
    from scripts.src.setup.managers.cache_manager import CacheManager
    from scripts.src.setup.managers.data_crawler import DataCrawler
    from scripts.src.setup.managers.data_processor import DataProcessor
    from scripts.src.setup.managers.system_tester import SystemTester

    __all__ = ["CacheManager", "DataCrawler", "DataProcessor", "SystemTester"]

except ImportError:
    __all__ = []

# Package metadata
__version__ = "1.0.0"
__description__ = "Management components for RAG system setup"
