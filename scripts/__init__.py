"""
Scripts package for RAG System Setup and Management.

This package contains CLI tools, setup orchestrators, and utilities
for managing the RAG (Retrieval-Augmented Generation) system.

Modules:
    - run: Main entry point for the setup system
    - src: Source code modules for setup functionality

Usage:
    From the command line:
        python scripts/run.py

    Or as a Python module:
        from scripts.run import cli_app, RAGSystemSetup
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
        StepTracker
    )
    
    __all__ = [
        'cli_app',
        'RAGSystemSetup',
        'SetupConfig',
        'SetupStep', 
        'EnvironmentValidator',
        'CacheManager',
        'DataCrawler',
        'DataProcessor',
        'SystemTester',
        'Logger',
        'StepTracker'
    ]
    
except ImportError:
    # If imports fail, provide minimal interface
    __all__ = []

# Package metadata
__version__ = '1.0.0'
__author__ = 'RAG System Development Team'
__description__ = 'Scripts and tools for RAG system setup and management'
