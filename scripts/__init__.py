"""
Scripts package untuk RAG system setup dan management

Package ini berisi CLI tools, setup orchestrators, dan utilities
untuk mengelola RAG (Retrieval-Augmented Generation) system.
"""

from pathlib import Path
import sys

# Add project root to path for proper imports
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Import main components
try:
    from scripts.run import (
        cli_app,
        RAGSystemSetup,
        SetupConfig,
        SetupStep,
        EnvironmentValidator,
        CacheManager,
        DataCrawler,
        DataProcessor,
        SystemTester,
        Logger,
        StepTracker,
    )

    __all__ = [
        "cli_app",
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
    # If imports fail, provide minimal interface
    __all__ = []

__version__ = "1.0.0"
