"""
User Interface package untuk RAG system setup

Package ini berisi UI components untuk displaying information, progress,
dan interactions selama RAG system setup process.
"""

try:
    from scripts.src.setup.ui.console import Console
    from scripts.src.setup.ui.display import Display

    __all__ = ["Console", "Display"]

except ImportError:
    __all__ = []

__version__ = "1.0.0"
__description__ = "User interface components for RAG system setup"
