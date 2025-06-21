"""
Setup package for RAG System.

This package contains all components needed for setting up and configuring
the RAG (Retrieval-Augmented Generation) system, including orchestration,
data management, validation, and user interface components.

Subpackages:
    - core: Core configuration, logging, and tracking utilities
    - managers: Data management and processing components
    - validators: Environment and configuration validation
    - ui: User interface and display components

Main Classes:
    - RAGSystemSetup: Main orchestrator for the setup process
    - SetupConfig: Configuration management
    - EnvironmentValidator: Environment validation
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

# Package metadata
__version__ = "1.0.0"
__description__ = "Setup orchestration and management for RAG system"
