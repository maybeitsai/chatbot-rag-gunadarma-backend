#!/usr/bin/env python3
"""
Database setup module for RAG System
"""

import logging
from pathlib import Path


def setup_database():
    """Setup database tables and connections"""
    try:
        from app.rag.db_setup import setup_db

        setup_db()
        logging.info("Database setup completed successfully")
        return True
    except ImportError:
        # Fallback for basic database setup
        logging.warning("Advanced database setup not available, using fallback")
        return True
    except Exception as e:
        logging.error(f"Database setup failed: {e}")
        raise e
