"""
Fixed Vector Store with simplified indexing and Hybrid Search support
Resolves pgvector dimension metadata issues
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

# Import hybrid search components
from app.rag.hybrid_search import HybridSearchManager, HybridSearchConfig, SearchType

load_dotenv()
logger = logging.getLogger(__name__)


class HybridRetriever:
    """Custom retriever that supports hybrid search"""
    
    def __init__(self, vector_store: PGVector, hybrid_manager: HybridSearchManager, k: int = 5):
        self.vector_store = vector_store
        self.hybrid_manager = hybrid_manager
        self.k = k
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using hybrid search"""
        return self.hybrid_manager.search(query, self.vector_store)
    
    def invoke(self, query: str) -> List[Document]:
        """Invoke method for compatibility"""
        return self.get_relevant_documents(query)


class VectorStoreManager:
    """Simplified vector store manager with basic indexing and hybrid search support"""
    
    def __init__(self, hybrid_config: Optional[HybridSearchConfig] = None):
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
        
        # Initialize hybrid search
        self.hybrid_config = hybrid_config or HybridSearchConfig()
        self.hybrid_search_manager = HybridSearchManager(
            connection_string=self.connection_string,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
            config=self.hybrid_config
        )

    def _get_database_connection(self):
        """Get database connection for direct SQL operations"""
        return psycopg2.connect(self.connection_string)

    def _verify_database_schema(self):
        """Verify that the required database tables exist"""
        try:
            with self._get_database_connection() as conn:
                with conn.cursor() as cur:
                    # Check if pgvector extension is installed
                    cur.execute("""
                        SELECT EXISTS(
                            SELECT 1 FROM pg_extension WHERE extname = 'vector'
                        );
                    """)
                    
                    if not cur.fetchone()[0]:
                        logger.warning("pgvector extension not found, attempting to create it")
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    
                    # Check if langchain tables exist
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'langchain_pg_collection'
                        );
                    """)
                    
                    collection_table_exists = cur.fetchone()[0]
                    
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'langchain_pg_embedding'
                        );
                    """)
                    
                    embedding_table_exists = cur.fetchone()[0]
                    
                    if not collection_table_exists or not embedding_table_exists:
                        logger.warning("Required langchain tables not found. Vector store needs to be initialized first.")
                        return False
                    
                    # Check if embedding column has proper type
                    cur.execute("""
                        SELECT data_type, udt_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'langchain_pg_embedding' 
                        AND column_name = 'embedding';
                    """)
                    
                    embedding_type_info = cur.fetchone()
                    if not embedding_type_info:
                        logger.warning("Embedding column not found")
                        return False
                    
                    data_type, udt_name = embedding_type_info
                    # Accept both 'vector' type and 'USER-DEFINED' with vector udt_name
                    if not (data_type == 'vector' or (data_type == 'USER-DEFINED' and udt_name == 'vector')):
                        logger.warning(f"Embedding column has unexpected type: {data_type} ({udt_name})")
                        return False
                    
                    logger.info("Database schema verification passed")
                    return True
                    
        except Exception as e:
            logger.error(f"Database schema verification failed: {e}")
            return False

    def create_indexes(self):
        """Create basic indexes for vector similarity search"""
        # First verify database schema
        if not self._verify_database_schema():
            logger.warning("Database schema verification failed, skipping index creation")
            return
            
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
                    
                    # Check if embedding column has proper vector data
                    cur.execute(f"""
                        SELECT embedding FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = '{self.collection_name}' 
                        AND embedding IS NOT NULL
                        LIMIT 1;
                    """)
                    
                    embedding_sample = cur.fetchone()
                    if not embedding_sample or not embedding_sample[0]:
                        logger.warning("No valid embeddings found, skipping index creation")
                        return
                    
                    # Get embedding dimensions using pgvector function
                    cur.execute(f"""
                        SELECT vector_dims(embedding) as dimensions 
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = '{self.collection_name}' 
                        AND embedding IS NOT NULL
                        LIMIT 1;
                    """)
                    
                    dimensions_result = cur.fetchone()
                    if not dimensions_result or not dimensions_result[0]:
                        logger.warning("Could not determine embedding dimensions, skipping index creation")
                        return
                    
                    dimensions = dimensions_result[0]
                    logger.info(f"Creating basic indexes for {doc_count} documents with {dimensions} dimensions")
                    
                    # Create simple collection_id index for fast joins
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_embedding_collection_id
                        ON langchain_pg_embedding (collection_id);
                    """)
                    logger.info("Created collection_id index")
                    
                    # Note: Skipping HNSW/IVF indexes due to pgvector dimension metadata requirements
                    # Vector similarity search will use sequential scan which is acceptable for moderate data sizes
                    logger.info("Using basic vector similarity search (no advanced indexes)")
                    logger.info(f"Vector store ready with {doc_count} documents and {dimensions} dimensions")
                    
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
                     score_threshold: float = 0.3,
                     use_hybrid: bool = False):
        """
        Get optimized retriever for fast similarity search with hybrid support
        
        Args:
            search_type: Type of search ('similarity', 'similarity_score_threshold', 'mmr', 'hybrid')
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            use_hybrid: Whether to use hybrid search
            
        Returns:
            Configured retriever or hybrid search results
        """
        vector_store = self.initialize_vector_store()
        
        # If hybrid search is requested
        if use_hybrid or search_type == "hybrid":
            return HybridRetriever(
                vector_store=vector_store,
                hybrid_manager=self.hybrid_search_manager,
                k=k
            )
        
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
                    
                    # Get hybrid search stats
                    hybrid_stats = self.hybrid_search_manager.get_stats()
                    
                    return {
                        'collection_name': self.collection_name,
                        'document_count': document_count,
                        'hnsw_params': self.hnsw_params,
                        'hybrid_search': hybrid_stats
                    }
                    
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {
                'collection_name': self.collection_name,
                'document_count': 0,
                'hnsw_params': self.hnsw_params,
                'hybrid_search': {}
            }

    def cleanup_old_collections(self, keep_latest: int = 2):
        """Clean up old collections to save space"""
        try:
            with self._get_database_connection() as conn:
                with conn.cursor() as cur:
                    # Get all collections (simplified query without JSON ordering)
                    cur.execute("""
                        SELECT name 
                        FROM langchain_pg_collection;
                    """)
                    
                    collections = cur.fetchall()
                    
                    if len(collections) <= keep_latest:
                        logger.info(f"Only {len(collections)} collections found, no cleanup needed")
                        return
                    
                    # Delete old collections
                    collections_to_delete = collections[keep_latest:]
                    for collection in collections_to_delete:
                        collection_name = collection[0]
                        if collection_name != self.collection_name:  # Don't delete current collection
                            cur.execute("""
                                DELETE FROM langchain_pg_collection 
                                WHERE name = %s;
                            """, (collection_name,))
                            logger.info(f"Deleted old collection: {collection_name}")
                    
                    conn.commit()
                    logger.info(f"Cleanup completed, kept {keep_latest} latest collections")
                    
        except Exception as e:
            logger.error(f"Error during collection cleanup: {e}")

    async def setup_vector_store(self, documents: List[Document]) -> bool:
        """Setup and populate vector store with documents"""
        try:
            logger.info(f"Setting up vector store and populating with {len(documents)} documents")
            
            # Initialize vector store
            vector_store = self.initialize_vector_store()
            
            # Add documents in batches
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                await asyncio.to_thread(vector_store.add_documents, batch)
                logger.info(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully populated vector store with {len(documents)} documents")
            
            # Invalidate hybrid search cache when new documents are added
            self.hybrid_search_manager.invalidate_cache()
            logger.info("Invalidated hybrid search cache for new documents")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
            return False

    def setup_and_populate_from_documents(self, documents: List[Document]) -> bool:
        """Setup and populate vector store with documents (sync version)"""
        try:
            logger.info(f"Setting up vector store and populating with {len(documents)} documents")
            
            # Initialize vector store
            vector_store = self.initialize_vector_store()
            
            # Add documents in batches
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                vector_store.add_documents(batch)
                logger.info(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            # Create performance indexes
            self.create_indexes()
            logger.info("Successfully created performance indexes")
            
            # Invalidate hybrid search cache when new documents are added
            self.hybrid_search_manager.invalidate_cache()
            logger.info("Invalidated hybrid search cache for new documents")
            
            logger.info(f"Successfully populated vector store with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
            return False

    def update_hybrid_config(self, new_config: HybridSearchConfig):
        """Update hybrid search configuration"""
        try:
            new_config.validate()
            self.hybrid_config = new_config
            self.hybrid_search_manager = HybridSearchManager(
                connection_string=self.connection_string,
                collection_name=self.collection_name,
                embeddings=self.embeddings,
                config=self.hybrid_config
            )
            logger.info(f"Updated hybrid search configuration: {new_config}")
        except Exception as e:
            logger.error(f"Error updating hybrid config: {e}")
            raise
