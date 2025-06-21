"""
Core utilities package for RAG System Setup.

This package contains essential utilities and configurations needed
for the setup process, including configuration management, logging,
step tracking, and enum definitions.

Modules:
    - config: Setup configuration management
    - enums: Enumeration definitions for setup steps and states
    - logger: Logging configuration and utilities
    - tracker: Step tracking and progress monitoring

These components provide the foundation for the setup orchestration process.
"""

try:
    from scripts.src.setup.core.config import SetupConfig
    from scripts.src.setup.core.enums import SetupStep
    from scripts.src.setup.core.logger import Logger
    from scripts.src.setup.core.tracker import StepTracker

    __all__ = ["SetupConfig", "SetupStep", "Logger", "StepTracker"]

except ImportError:
    __all__ = []

# Package metadata
__version__ = "1.0.0"
__description__ = "Core utilities for RAG system setup"
