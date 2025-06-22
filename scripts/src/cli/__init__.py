"""
Command-Line Interface (CLI) package untuk RAG system

Package ini menyediakan CLI tools dan commands untuk interaksi dengan
RAG system setup dan management functionality.
"""

try:
    from scripts.src.cli.app import app

    __all__ = ["app"]

except ImportError:
    __all__ = []

__version__ = "1.0.0"
