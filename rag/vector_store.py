"""
Final optimized Vector Store with HNSW indexing
Simplified version without metadata filtering to avoid subquery errors
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Optimized vector store with HNSW indexing for fast similarity search"""
    
    def __init__(self):
        self.connection_string = os.getenv("NEON_CONNECTION_STRING")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model,
            google_api_key=self.google_api_key
        )
        
        # Collection name for PGVector
        self.collection_name = "chatbot_gunadarma"
        
        # HNSW parameters for optimal performance
        self.hnsw_params = {
            "m": 16,  # Number of bi-directional links for each node
            "ef_construction": 64,  # Size of candidate list during index construction
            "ef_search": 40  # Size of candidate list during search
        }

    def _get_database_connection(self):
        """Get database connection for direct SQL operations"""
        return psycopg2.connect(self.connection_string)

    def create_indexes(self):
        """Create HNSW index for optimal vector similarity search"""
        try:
            with self._get_database_connection() as conn:
                with conn.cursor() as cur:
                    # Enable pgvector extension if not exists
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    
                    # Check if collection exists and has documents
                    cur.execute(f"""
                        SELECT COUNT(*) FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = '{self.collection_name}';
                    """)
                    
                    doc_count = cur.fetchone()[0]
                    
                    if doc_count == 0:
                        logger.info("No documents found, skipping index creation")
                        return
                    
                    logger.info(f"Creating indexes for {doc_count} documents")
                    
                    # Create simple collection_id index for fast joins
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_embedding_collection_id
                        ON langchain_pg_embedding (collection_id);
                    """)
                    logger.info("Created collection_id index")
                    
                    # Try to create HNSW index (optional - will warn if fails)
                    index_name = f"hnsw_idx_{self.collection_name}_embedding"
                    try:
                        cur.execute(f"""
                            CREATE INDEX IF NOT EXISTS {index_name}
                            ON langchain_pg_embedding
                            USING hnsw (embedding vector_cosine_ops)
                            WITH (m = {self.hnsw_params['m']}, ef_construction = {self.hnsw_params['ef_construction']});
                        """)
                        logger.info(f"Created HNSW index: {index_name}")
                    except Exception as hnsw_error:
                        logger.warning(f"HNSW index creation failed (this is optional): {hnsw_error}")
                        # Try IVF index as fallback
                        try:
                            ivf_index_name = f"ivf_idx_{self.collection_name}_embedding"
                            cur.execute(f"""
                                CREATE INDEX IF NOT EXISTS {ivf_index_name}
                                ON langchain_pg_embedding
                                USING ivfflat (embedding vector_cosine_ops)
                                WITH (lists = 100);
                            """)
                            logger.info(f"Created IVF index as fallback: {ivf_index_name}")
                        except Exception as ivf_error:
                            logger.warning(f"IVF index also failed (using brute force search): {ivf_error}")
                    
                    # Set HNSW search parameters if available
                    try:
                        cur.execute(f"SET hnsw.ef_search = {self.hnsw_params['ef_search']};")
                    except:
                        pass  # Ignore if HNSW not available
                    
                    conn.commit()
                    logger.info("Index creation completed successfully")
                    
        except Exception as e:
            logger.warning(f"Index creation had issues (vector store will still work): {e}")

    def initialize_vector_store(self) -> PGVector:
        """Initialize PGVector store"""
        try:
            vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self.connection_string,
                use_jsonb=True,
                pre_delete_collection=False,
                logger=logger
            )
            
            logger.info(f"Vector store initialized successfully for collection: {self.collection_name}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def get_retriever(self, 
                     search_type: str = "similarity_score_threshold",
                     k: int = 5,
                     score_threshold: float = 0.3):
        """
        Get optimized retriever for fast similarity search
        
        Args:
            search_type: Type of search ('similarity', 'similarity_score_threshold', 'mmr')
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            
        Returns:
            Configured retriever
        """
        vector_store = self.initialize_vector_store()
        
        search_kwargs = {"k": k}
        
        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold
        
        return vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            with self._get_database_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get collection info
                    cur.execute("""
                        SELECT COUNT(*) as document_count
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %s
                    """, (self.collection_name,))
                    
                    result = cur.fetchone()
                    document_count = result['document_count'] if result else 0
                    
                    return {
                        'collection_name': self.collection_name,
                        'document_count': document_count,
                        'hnsw_params': self.hnsw_params
                    }
                    
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {
                'error': str(e),
                'collection_name': self.collection_name
            }

    def cleanup_old_collections(self, keep_latest: int = 2):
        """
        Clean up old collections to save space
        
        Args:
            keep_latest: Number of latest collections to keep
        """
        try:
            with self._get_database_connection() as conn:
                with conn.cursor() as cur:
                    # Get all collections sorted by creation time
                    cur.execute("""
                        SELECT name, uuid, cmetadata
                        FROM langchain_pg_collection
                        WHERE name LIKE 'chatbot_gunadarma%'
                        ORDER BY (cmetadata->>'created_at')::timestamp DESC
                    """)
                    
                    collections = cur.fetchall()
                    
                    if len(collections) > keep_latest:
                        collections_to_delete = collections[keep_latest:]
                        
                        for collection in collections_to_delete:
                            collection_uuid = collection[1]
                            collection_name = collection[0]
                            
                            # Delete embeddings first
                            cur.execute("""
                                DELETE FROM langchain_pg_embedding 
                                WHERE collection_id = %s
                            """, (collection_uuid,))
                            
                            # Delete collection
                            cur.execute("""
                                DELETE FROM langchain_pg_collection 
                                WHERE uuid = %s
                            """, (collection_uuid,))
                            
                            logger.info(f"Cleaned up old collection: {collection_name}")
                        
                        conn.commit()
                        logger.info(f"Cleaned up {len(collections_to_delete)} old collections")
                    else:
                        logger.info(f"Only {len(collections)} collections found, no cleanup needed")
                        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def setup_and_populate_from_documents(self, documents: List[Document], batch_size: int = 50):
        """
        Setup vector store and populate with documents in batches
        
        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process in each batch
        """
        try:
            if not documents:
                logger.warning("No documents provided for population")
                return
                
            logger.info(f"Setting up vector store and populating with {len(documents)} documents")
            
            # Initialize vector store
            vector_store = self.initialize_vector_store()
            
            # Add documents in batches to avoid memory issues
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                try:
                    vector_store.add_documents(batch)
                    logger.info(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                except Exception as e:
                    logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                    continue
            
            # Create indexes for performance
            try:
                self.create_indexes()
                logger.info("Successfully created performance indexes")
            except Exception as index_error:
                logger.warning(f"Could not create indexes (this is optional): {index_error}")
                logger.info("Vector store will work without indexes, just with reduced performance")
            
            logger.info(f"Successfully populated vector store with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error in setup_and_populate_from_documents: {e}")
            raise

if __name__ == "__main__":
    # Test the vector store
    logging.basicConfig(level=logging.INFO)
    
    manager = VectorStoreManager()
    
    # Get stats
    stats = manager.get_vector_store_stats()
    print(f"Vector store stats: {stats}")
    
    # Test simple similarity search
    retriever = manager.get_retriever(k=3)
    print("Vector store initialized successfully")
