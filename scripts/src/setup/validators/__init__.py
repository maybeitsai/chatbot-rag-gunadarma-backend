"""
Validators package for RAG System Setup.

This package contains validation components that ensure the system
environment and configuration are properly set up before proceeding
with the RAG system installation and configuration.

Modules:
    - environment: Environment validation and dependency checking

The validators help prevent setup failures by checking prerequisites
and system requirements early in the setup process.
"""

try:
    from scripts.src.setup.validators.environment import EnvironmentValidator

    __all__ = ["EnvironmentValidator"]

except ImportError:
    __all__ = []

# Package metadata
__version__ = "1.0.0"
__description__ = "Validation components for RAG system setup"
