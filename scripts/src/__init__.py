"""
Source code modules untuk RAG system scripts

Package ini berisi core functionality untuk setup dan management
RAG (Retrieval-Augmented Generation) system.
"""

# Re-export main components for convenience
try:
    from scripts.src.cli.app import app as cli_app
    from scripts.src.setup.orchestrator import RAGSystemSetup

    __all__ = ["cli_app", "RAGSystemSetup"]

except ImportError:
    # Graceful degradation if imports fail
    __all__ = []

__version__ = "1.0.0"
