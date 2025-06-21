"""
Source code modules for RAG System Scripts.

This package contains the core functionality for setting up and managing
the RAG (Retrieval-Augmented Generation) system, including CLI interfaces,
setup orchestration, and various utility modules.

Subpackages:
    - cli: Command-line interface components
    - setup: Setup orchestration and management modules

The main entry point is typically through the CLI app or the setup orchestrator.
"""

# Re-export main components for convenience
try:
    from scripts.src.cli.app import app as cli_app
    from scripts.src.setup.orchestrator import RAGSystemSetup
    
    __all__ = [
        'cli_app',
        'RAGSystemSetup'
    ]
    
except ImportError:
    # Graceful degradation if imports fail
    __all__ = []

# Package metadata
__version__ = '1.0.0'
__description__ = 'Source modules for RAG system scripts'