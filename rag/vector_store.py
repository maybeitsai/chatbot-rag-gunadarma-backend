"""
Optimized Vector Store with HNSW indexing and metadata filtering
Enhanced performance for large-scale document retrieval
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
    """Enhanced vector store with HNSW indexing and metadata filtering"""
    
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
        """Create HNSW index and metadata indexes for optimal performance"""
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
                    
                    # Get embedding dimension from existing data
                    cur.execute(f"""
                        SELECT array_length(embedding, 1) as dim 
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = '{self.collection_name}'
                        AND embedding IS NOT NULL
                        LIMIT 1;
                    """)
                    
                    result = cur.fetchone()
                    if not result or not result[0]:
                        logger.warning("No valid embeddings found, skipping HNSW index creation")
                        return
                    
                    embedding_dim = result[0]
                    logger.info(f"Creating HNSW index for embeddings with dimension {embedding_dim}")
                    
                    # Create HNSW index on embedding column only if embeddings exist
                    index_name = f"hnsw_idx_{self.collection_name}_embedding"
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON langchain_pg_embedding
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = {self.hnsw_params['m']}, ef_construction = {self.hnsw_params['ef_construction']})
                        WHERE collection_id = (
                            SELECT uuid FROM langchain_pg_collection 
                            WHERE name = '{self.collection_name}'
                        );
                    """)
                    
                    # Create metadata indexes for filtering
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_url
                        ON langchain_pg_embedding
                        USING btree ((cmetadata->>'url'))
                        WHERE collection_id = (
                            SELECT uuid FROM langchain_pg_collection 
                            WHERE name = '{self.collection_name}'
                        );
                    """)
                    
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_source_type
                        ON langchain_pg_embedding
                        USING btree ((cmetadata->>'source_type'))
                        WHERE collection_id = (
                            SELECT uuid FROM langchain_pg_collection 
                            WHERE name = '{self.collection_name}'
                        );
                    """)
                    
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_content_length
                        ON langchain_pg_embedding
                        USING btree (((cmetadata->>'content_length')::int))
                        WHERE collection_id = (
                            SELECT uuid FROM langchain_pg_collection 
                            WHERE name = '{self.collection_name}'
                        );
                    """)
                    
                    # Set HNSW search parameters
                    cur.execute(f"SET hnsw.ef_search = {self.hnsw_params['ef_search']};")
                    
                    conn.commit()
                    logger.info("Optimized indexes created successfully")
                    
        except Exception as e:
            logger.error(f"Error creating optimized indexes: {e}")
            raise
    
    def initialize_vector_store(self) -> PGVector:
        """Initialize optimized PGVector store"""
        try:
            vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self.connection_string,
                use_jsonb=True,
                pre_delete_collection=False,
                logger=logger
            )
            
            # Create optimized indexes after vector store initialization
            self.create_indexes()
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    async def add_documents_batch(self, documents: List[Document], batch_size: int = 50):
        """
        Add documents to vector store in optimized batches
        
        Args:
            documents: List of documents to add
            batch_size: Size of each batch for processing
        """
        vector_store = self.initialize_vector_store()
        
        logger.info(f"Adding {len(documents)} documents in batches of {batch_size}")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                # Add documents to vector store
                await asyncio.to_thread(vector_store.add_documents, batch)
                logger.info(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                
                # Small delay to prevent overwhelming the database
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                # Continue with next batch instead of failing completely
                continue
    
    def get_retriever(self, 
                               search_type: str = "similarity_score_threshold",
                               k: int = 5,
                               score_threshold: float = 0.3,
                               metadata_filter: Optional[Dict[str, Any]] = None):
        """
        Get optimized retriever with metadata filtering
        
        Args:
            search_type: Type of search ('similarity', 'similarity_score_threshold', 'mmr')
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            metadata_filter: Dictionary of metadata filters
            
        Returns:
            Configured retriever
        """
        vector_store = self.initialize_vector_store()
        
        search_kwargs = {"k": k}
        
        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold
        
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        
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
                      # Get metadata statistics
                    cur.execute("""
                        SELECT 
                            COUNT(DISTINCT e.cmetadata->>'url') as unique_urls,
                            COUNT(DISTINCT e.cmetadata->>'source_type') as source_types,
                            AVG((e.cmetadata->>'content_length')::int) as avg_content_length,
                            MIN((e.cmetadata->>'content_length')::int) as min_content_length,
                            MAX((e.cmetadata->>'content_length')::int) as max_content_length
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %s
                        AND e.cmetadata->>'content_length' IS NOT NULL
                    """, (self.collection_name,))
                    
                    stats_result = cur.fetchone()
                    
                    return {
                        'collection_name': self.collection_name,
                        'document_count': document_count,
                        'unique_urls': stats_result['unique_urls'] if stats_result else 0,
                        'source_types': stats_result['source_types'] if stats_result else 0,
                        'avg_content_length': round(stats_result['avg_content_length'], 2) if stats_result and stats_result['avg_content_length'] else 0,
                        'min_content_length': stats_result['min_content_length'] if stats_result else 0,
                        'max_content_length': stats_result['max_content_length'] if stats_result else 0,
                        'hnsw_params': self.hnsw_params
                    }
                    
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {
                'error': str(e),
                'collection_name': self.collection_name
            }
    
    def search_with_metadata_filter(self, 
                                   query: str, 
                                   metadata_filter: Dict[str, Any], 
                                   k: int = 5,
                                   score_threshold: float = 0.5) -> List[Document]:
        """
        Search documents with metadata filtering for better precision
        
        Args:
            query: Search query
            metadata_filter: Metadata filters to apply
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of matching documents
        """
        try:
            retriever = self.get_retriever(
                search_type="similarity_score_threshold",
                k=k,
                score_threshold=score_threshold,
                metadata_filter=metadata_filter
            )
            
            results = retriever.get_relevant_documents(query)
            logger.info(f"Found {len(results)} documents with metadata filter: {metadata_filter}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in metadata filtered search: {e}")
            return []
    
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
                            
                            logger.info(f"Deleted old collection: {collection_name}")
                    
                    conn.commit()
                    logger.info(f"Cleanup completed, kept {keep_latest} latest collections")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    # Test the vector store
    logging.basicConfig(level=logging.INFO)
    
    manager = VectorStoreManager()
    
    # Get stats
    stats = manager.get_vector_store_stats()
    print(f"Vector store stats: {stats}")
    
    # Test metadata filtering
    metadata_filter = {"source_type": "html"}
    results = manager.search_with_metadata_filter(
        query="Universitas Gunadarma",
        metadata_filter=metadata_filter,
        k=3
    )
    
    print(f"Found {len(results)} documents with metadata filter")
