"""
User Interface package for RAG System Setup.

This package contains UI components for displaying information,
progress, and interactions during the RAG system setup process.

Modules:
    - console: Console-based user interface utilities
    - display: Display components for setup progress and information

These components provide user-friendly interfaces for monitoring
and interacting with the setup process.
"""

try:
    from scripts.src.setup.ui.console import Console
    from scripts.src.setup.ui.display import Display

    __all__ = ["Console", "Display"]

except ImportError:
    __all__ = []

# Package metadata
__version__ = "1.0.0"
__description__ = "User interface components for RAG system setup"
