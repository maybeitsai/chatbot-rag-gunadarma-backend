#!/usr/bin/env python3
"""
Vector store management for RAG System
"""

import logging
from typing import Dict, Any

class VectorStoreManager:
    """Vector store management and optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def cleanup_old_collections(self, keep_latest: int = 2):
        """Cleanup old vector store collections"""
        try:
            from app.rag.vector_store import VectorStoreManager as AppVectorStoreManager
            app_manager = AppVectorStoreManager()
            
            if hasattr(app_manager, 'cleanup_old_collections'):
                app_manager.cleanup_old_collections(keep_latest)
                self.logger.info(f"Cleaned up old collections, keeping {keep_latest} latest")
            else:
                self.logger.warning("Cleanup method not available in app vector store manager")
                
        except ImportError:
            self.logger.warning("App vector store manager not available for cleanup")
        except Exception as e:
            self.logger.error(f"Vector store cleanup failed: {e}")
    
    def create_indexes(self):
        """Create performance indexes for vector store"""
        try:
            from app.rag.vector_store import VectorStoreManager as AppVectorStoreManager
            app_manager = AppVectorStoreManager()
            
            if hasattr(app_manager, 'create_indexes'):
                app_manager.create_indexes()
                self.logger.info("Vector store indexes created successfully")
            else:
                self.logger.warning("Index creation method not available")
                
        except ImportError:
            self.logger.warning("App vector store manager not available for index creation")
        except Exception as e:
            self.logger.error(f"Vector store index creation failed: {e}")
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            from app.rag.vector_store import VectorStoreManager as AppVectorStoreManager
            app_manager = AppVectorStoreManager()
            
            if hasattr(app_manager, 'get_vector_store_stats'):
                return app_manager.get_vector_store_stats()
            else:
                # Fallback stats
                return {
                    'document_count': 0,
                    'collection_count': 0,
                    'status': 'unknown'
                }
                
        except ImportError:
            return {
                'document_count': 0,
                'collection_count': 0,
                'status': 'not_available'
            }
        except Exception as e:
            self.logger.error(f"Failed to get vector store stats: {e}")
            return {
                'document_count': 0,
                'collection_count': 0,
                'status': 'error'
            }
