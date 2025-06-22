"""
Setup package untuk RAG system

Package ini berisi semua komponen yang diperlukan untuk setup dan konfigurasi
RAG (Retrieval-Augmented Generation) system.
"""

try:
    # Import main orchestrator and key components
    from scripts.src.setup.orchestrator import RAGSystemSetup
    from scripts.src.setup.core.config import SetupConfig
    from scripts.src.setup.core.enums import SetupStep
    from scripts.src.setup.validators.environment import EnvironmentValidator

    # Import managers
    from scripts.src.setup.managers.cache_manager import CacheManager
    from scripts.src.setup.managers.data_crawler import DataCrawler
    from scripts.src.setup.managers.data_processor import DataProcessor
    from scripts.src.setup.managers.system_tester import SystemTester

    # Import core utilities
    from scripts.src.setup.core.logger import Logger
    from scripts.src.setup.core.tracker import StepTracker

    __all__ = [
        "RAGSystemSetup",
        "SetupConfig",
        "SetupStep",
        "EnvironmentValidator",
        "CacheManager",
        "DataCrawler",
        "DataProcessor",
        "SystemTester",
        "Logger",
        "StepTracker",
    ]

except ImportError:
    __all__ = []

__version__ = "1.0.0"
