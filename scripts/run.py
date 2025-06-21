#!/usr/bin/env python3
"""
Unified Setup Script for RAG System - Entry point
"""

import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from scripts.src.cli.app import app as cli_app

# Import classes from their proper modules
from scripts.src.setup.orchestrator import RAGSystemSetup
from scripts.src.setup.core.config import SetupConfig
from scripts.src.setup.core.enums import SetupStep
from scripts.src.setup.core.logger import Logger
from scripts.src.setup.core.tracker import StepTracker
from scripts.src.setup.managers.cache_manager import CacheManager
from scripts.src.setup.managers.data_crawler import DataCrawler
from scripts.src.setup.managers.data_processor import DataProcessor
from scripts.src.setup.managers.system_tester import SystemTester
from scripts.src.setup.validators.environment import EnvironmentValidator

# Expose classes for direct import
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

if __name__ == "__main__":
    cli_app()