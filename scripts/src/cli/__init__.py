"""
Command-Line Interface (CLI) package for RAG System.

This package provides CLI tools and commands for interacting with
the RAG system setup and management functionality.

Modules:
    - app: Main CLI application using Typer
    - commands: Individual CLI command implementations

Usage:
    The main CLI app can be accessed via:
        from scripts.src.cli import app

    Or run directly:
        python -m scripts.src.cli.app
"""

try:
    from scripts.src.cli.app import app

    __all__ = ["app"]

except ImportError:
    __all__ = []

# Package metadata
__version__ = "1.0.0"
__description__ = "CLI tools for RAG system management"
