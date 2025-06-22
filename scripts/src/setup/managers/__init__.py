"""
Managers package untuk RAG system setup

Package ini berisi management components yang bertanggung jawab untuk
berbagai aspek dari RAG system setup process.
"""

try:
    from scripts.src.setup.managers.cache_manager import CacheManager
    from scripts.src.setup.managers.data_crawler import DataCrawler
    from scripts.src.setup.managers.data_processor import DataProcessor
    from scripts.src.setup.managers.system_tester import SystemTester

    __all__ = ["CacheManager", "DataCrawler", "DataProcessor", "SystemTester"]

except ImportError:
    __all__ = []

__version__ = "1.0.0"
__description__ = "Management components for RAG system setup"
