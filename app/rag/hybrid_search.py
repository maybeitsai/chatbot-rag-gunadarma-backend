"""
Hybrid Search Configuration and Implementation
Combines dense vector search with sparse keyword search (BM25)
Based on LangChain hybrid search patterns
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
from langchain.schema import Document
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

logger = logging.getLogger(__name__)


def create_indonesian_stopwords() -> Set[str]:
    """
    Buat daftar stopwords bahasa Indonesia menggunakan Sastrawi 
    dan tambahan stopwords custom
    """
    try:
        # Gunakan Sastrawi untuk stopwords bahasa Indonesia
        factory = StopWordRemoverFactory()
        all_stopwords = set(factory.get_stop_words())
        
        logger.info(f"Created Indonesian stopwords list with {len(all_stopwords)} words")
        return all_stopwords
        
    except Exception as e:
        logger.warning(f"Error creating Indonesian stopwords: {e}, using basic stopwords")
        # Fallback ke stopwords dasar jika Sastrawi gagal
        basic_stopwords = {
            'dan', 'atau', 'yang', 'adalah', 'ini', 'itu', 'dari', 'ke', 'dengan',
            'untuk', 'pada', 'dalam', 'akan', 'telah', 'sudah', 'tidak', 'juga',
            'dapat', 'bisa', 'ada', 'saya', 'anda', 'dia', 'mereka', 'kita',
            'kami', 'nya', 'ku', 'mu', 'di', 'ke', 'dari', 'oleh', 'sebagai'
        }
        return basic_stopwords


class SearchType(Enum):
    """Supported search types"""
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    HYBRID_RRF = "hybrid_rrf"


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search"""
    # Search type
    search_type: SearchType = SearchType.HYBRID_RRF
    
    # Hybrid search weights (must sum to 1.0)
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    
    # Result configuration
    k: int = 5
    keyword_k: int = 16
    vector_k: int = 16
    
    # Scoring thresholds
    vector_score_threshold: float = 0.20
    keyword_score_threshold: float = 0.12
    hybrid_score_threshold: float = 0.16
    
    # RRF parameters
    rrf_k: int = 60

    # TF-IDF parameters
    tfidf_max_features: int = 10000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_min_df: int = 1
    tfidf_max_df: float = 0.8

    def validate(self) -> bool:
        """Validate configuration"""
        if abs(self.vector_weight + self.keyword_weight - 1.0) > 0.001:
            raise ValueError("vector_weight + keyword_weight must equal 1.0")
        
        if not (0.0 <= self.vector_weight <= 1.0):
            raise ValueError("vector_weight must be between 0.0 and 1.0")
        
        if not (0.0 <= self.keyword_weight <= 1.0):
            raise ValueError("keyword_weight must be between 0.0 and 1.0")
        
        return True


class HybridSearchManager:
    """Manages hybrid search functionality"""
    
    def __init__(self, connection_string: str, collection_name: str, 
                 embeddings: GoogleGenerativeAIEmbeddings,
                 config: Optional[HybridSearchConfig] = None):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.config = config or HybridSearchConfig()
        self.config.validate()
        
        # Initialize Indonesian stopwords
        try:
            self.stopwords = create_indonesian_stopwords()
        except Exception as e:
            logger.warning(f"Failed to create stopwords: {e}, using basic set")
            self.stopwords = {'dan', 'atau', 'yang', 'adalah', 'ini', 'itu', 'dari', 'ke', 'dengan'}
        
        # TF-IDF vectorizer for keyword search
        self.tfidf_vectorizer = None
        self.document_texts = []
        self.document_metadata = []
        
        # Cache paths
        self.cache_dir = "cache"
        self.tfidf_cache_path = os.path.join(self.cache_dir, "tfidf_vectorizer.pkl")
        self.texts_cache_path = os.path.join(self.cache_dir, "document_texts.json")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Hybrid search initialized with config: {self.config}")

    def _get_database_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.connection_string)

    def _load_document_texts(self) -> bool:
        """Load document texts from database and cache"""
        try:
            # Try to load from cache first
            if os.path.exists(self.texts_cache_path):
                try:
                    with open(self.texts_cache_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                        self.document_texts = cache_data.get('texts', [])
                        self.document_metadata = cache_data.get('metadata', [])
                        
                    logger.info(f"Loaded {len(self.document_texts)} document texts from cache")
                    
                    # Validate cache data
                    if len(self.document_texts) == 0:
                        logger.warning("Cache contains no document texts, will reload from database")
                        # Remove invalid cache and reload from database
                        os.remove(self.texts_cache_path)
                    else:
                        return True
                except Exception as cache_error:
                    logger.error(f"Error loading from cache: {cache_error}")
                    # Remove corrupted cache and reload from database
                    if os.path.exists(self.texts_cache_path):
                        os.remove(self.texts_cache_path)
            
            # Load from database
            logger.info(f"Loading document texts from database for collection: {self.collection_name}")
            with self._get_database_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # First check if collection exists
                    cur.execute("""
                        SELECT uuid, name FROM langchain_pg_collection WHERE name = %s
                    """, (self.collection_name,))
                    
                    collection_result = cur.fetchone()
                    if not collection_result:
                        logger.error(f"Collection '{self.collection_name}' not found in database")
                        return False
                    
                    logger.info(f"Found collection: {collection_result['name']} with UUID: {collection_result['uuid']}")
                    
                    # Now get documents
                    cur.execute("""
                        SELECT e.document, e.cmetadata
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %s
                        ORDER BY e.id
                    """, (self.collection_name,))
                    
                    results = cur.fetchall()
                    logger.info(f"Found {len(results)} documents in database")
                    
                    self.document_texts = []
                    self.document_metadata = []
                    
                    for i, row in enumerate(results):
                        doc_text = row['document']
                        if doc_text and doc_text.strip():
                            self.document_texts.append(doc_text)
                            self.document_metadata.append(row['cmetadata'] or {})
                        else:
                            logger.warning(f"Skipping empty document at index {i}")
            
            logger.info(f"Processed {len(self.document_texts)} valid document texts from {len(results)} total documents")
            
            # Only cache if we have valid documents
            if len(self.document_texts) > 0:
                cache_data = {
                    'texts': self.document_texts,
                    'metadata': self.document_metadata
                }
                with open(self.texts_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Cached {len(self.document_texts)} document texts")
            else:
                logger.warning("No valid document texts found to cache")
            
            return len(self.document_texts) > 0
            
        except Exception as e:
            logger.error(f"Error loading document texts: {e}")
            return False

    def _initialize_tfidf_vectorizer(self) -> bool:
        """Initialize and fit TF-IDF vectorizer"""
        try:
            # Try to load from cache first
            if os.path.exists(self.tfidf_cache_path):
                with open(self.tfidf_cache_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info("Loaded TF-IDF vectorizer from cache")
                return True
            
            # Ensure we have document texts
            if not self.document_texts and not self._load_document_texts():
                logger.error("No document texts available for TF-IDF training")
                return False
            
            # Validate document texts
            if len(self.document_texts) == 0:
                logger.error("Document texts list is empty, cannot train TF-IDF vectorizer")
                return False
            
            # Filter out empty documents
            valid_texts = [text for text in self.document_texts if text and text.strip()]
            if len(valid_texts) == 0:
                logger.error("All document texts are empty, cannot train TF-IDF vectorizer")
                return False
            
            if len(valid_texts) < len(self.document_texts):
                logger.warning(f"Found {len(self.document_texts) - len(valid_texts)} empty documents, using {len(valid_texts)} valid texts")
                self.document_texts = valid_texts
            
            # Initialize and fit vectorizer dengan stopwords bahasa Indonesia
            # Pastikan stopwords sudah ada, jika tidak buat default
            if not hasattr(self, 'stopwords') or not self.stopwords:
                try:
                    self.stopwords = create_indonesian_stopwords()
                except Exception as e:
                    logger.warning(f"Failed to create stopwords: {e}, using basic set")
                    self.stopwords = {'dan', 'atau', 'yang', 'adalah', 'ini', 'itu', 'dari', 'ke', 'dengan'}
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                ngram_range=self.config.tfidf_ngram_range,
                min_df=self.config.tfidf_min_df,
                max_df=self.config.tfidf_max_df,
                stop_words=list(self.stopwords),
                lowercase=True
            )
            
            # Fit the vectorizer
            self.tfidf_vectorizer.fit(self.document_texts)
            
            # Validate that the vectorizer was trained successfully
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_') or len(self.tfidf_vectorizer.vocabulary_) == 0:
                logger.error("TF-IDF vectorizer training failed - no vocabulary created")
                self.tfidf_vectorizer = None
                return False
            
            # Cache the vectorizer
            with open(self.tfidf_cache_path, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            logger.info(f"Trained and cached TF-IDF vectorizer with {len(self.tfidf_vectorizer.vocabulary_)} features")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing TF-IDF vectorizer: {e}")
            self.tfidf_vectorizer = None
            return False

    def _keyword_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Perform keyword search using TF-IDF"""
        try:
            if not self.tfidf_vectorizer:
                if not self._initialize_tfidf_vectorizer():
                    logger.warning("TF-IDF vectorizer initialization failed, skipping keyword search")
                    return []
            
            # Ensure we have document texts
            if not self.document_texts:
                if not self._load_document_texts():
                    logger.warning("No document texts available for keyword search")
                    return []
            
            # Check if we have enough documents
            if len(self.document_texts) == 0:
                logger.warning("Document texts is empty, cannot perform keyword search")
                return []
            
            # Validate that query is not empty
            if not query or not query.strip():
                logger.warning("Empty query provided for keyword search")
                return []
            
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Transform all documents
            doc_vectors = self.tfidf_vectorizer.transform(self.document_texts)
            
            # Check if we got valid vectors
            if query_vector.shape[0] == 0 or doc_vectors.shape[0] == 0:
                logger.warning("Empty vectors after TF-IDF transformation")
                return []
            
            # Calculate cosine similarity
            similarities = (doc_vectors * query_vector.T).toarray().flatten()
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            # Filter by threshold and return with scores
            results = []
            for idx in top_indices:
                if idx < len(similarities):
                    score = similarities[idx]
                    if score >= self.config.keyword_score_threshold:
                        results.append((int(idx), float(score)))
            
            logger.debug(f"Keyword search found {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

    def _vector_search(self, query: str, vector_store: PGVector, k: int) -> List[Tuple[Document, float]]:
        """Perform vector similarity search"""
        try:
            # Use similarity search with score
            results = vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter by threshold
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= self.config.vector_score_threshold
            ]
            
            logger.debug(f"Vector search found {len(filtered_results)} results for query: {query[:50]}...")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    def _apply_reciprocal_rank_fusion(self, 
                                     vector_results: List[Tuple[Document, float]], 
                                     keyword_results: List[Tuple[int, float]]) -> List[Tuple[Document, float]]:
        """Apply Reciprocal Rank Fusion to combine results"""
        try:
            # Create document index mapping for keyword results
            doc_scores = {}
            
            # Process vector results
            for rank, (doc, score) in enumerate(vector_results):
                doc_id = str(doc.page_content)
                rrf_score = 1.0 / (self.config.rrf_k + rank + 1)
                doc_scores[doc_id] = {
                    'document': doc,
                    'vector_rrf': rrf_score,
                    'keyword_rrf': 0.0,
                    'vector_score': score
                }
            
            # Process keyword results
            for rank, (doc_idx, score) in enumerate(keyword_results):
                if doc_idx < len(self.document_texts):
                    doc_text = self.document_texts[doc_idx]
                    doc_metadata = self.document_metadata[doc_idx] if doc_idx < len(self.document_metadata) else {}
                    
                    # Create document object
                    doc = Document(page_content=doc_text, metadata=doc_metadata)
                    doc_id = str(doc.page_content)
                    
                    rrf_score = 1.0 / (self.config.rrf_k + rank + 1)
                    
                    if doc_id in doc_scores:
                        doc_scores[doc_id]['keyword_rrf'] = rrf_score
                    else:
                        doc_scores[doc_id] = {
                            'document': doc,
                            'vector_rrf': 0.0,
                            'keyword_rrf': rrf_score,
                            'vector_score': 0.0
                        }
            
            # Calculate final RRF scores
            final_results = []
            for doc_id, scores in doc_scores.items():
                final_score = scores['vector_rrf'] + scores['keyword_rrf']
                final_results.append((scores['document'], final_score))
            
            # Sort by final score and return top k
            final_results.sort(key=lambda x: x[1], reverse=True)
            return final_results[:self.config.k]
            
        except Exception as e:
            logger.error(f"Error in RRF fusion: {e}")
            return vector_results[:self.config.k]

    def _apply_weighted_fusion(self, 
                              vector_results: List[Tuple[Document, float]], 
                              keyword_results: List[Tuple[int, float]]) -> List[Tuple[Document, float]]:
        """Apply weighted fusion to combine results"""
        try:
            doc_scores = {}
            
            # Normalize scores to 0-1 range
            if vector_results:
                max_vector_score = max(score for _, score in vector_results)
                min_vector_score = min(score for _, score in vector_results)
                vector_range = max_vector_score - min_vector_score if max_vector_score > min_vector_score else 1.0
            
            if keyword_results:
                max_keyword_score = max(score for _, score in keyword_results)
                min_keyword_score = min(score for _, score in keyword_results)
                keyword_range = max_keyword_score - min_keyword_score if max_keyword_score > min_keyword_score else 1.0
            
            # Process vector results
            for doc, score in vector_results:
                doc_id = str(doc.page_content)
                normalized_score = (score - min_vector_score) / vector_range if vector_results else 0.0
                weighted_score = normalized_score * self.config.vector_weight
                
                doc_scores[doc_id] = {
                    'document': doc,
                    'final_score': weighted_score
                }
            
            # Process keyword results
            for doc_idx, score in keyword_results:
                if doc_idx < len(self.document_texts):
                    doc_text = self.document_texts[doc_idx]
                    doc_metadata = self.document_metadata[doc_idx] if doc_idx < len(self.document_metadata) else {}
                    
                    doc = Document(page_content=doc_text, metadata=doc_metadata)
                    doc_id = str(doc.page_content)
                    
                    normalized_score = (score - min_keyword_score) / keyword_range if keyword_results else 0.0
                    weighted_score = normalized_score * self.config.keyword_weight
                    
                    if doc_id in doc_scores:
                        doc_scores[doc_id]['final_score'] += weighted_score
                    else:
                        doc_scores[doc_id] = {
                            'document': doc,
                            'final_score': weighted_score
                        }
            
            # Convert to list and sort
            final_results = [
                (scores['document'], scores['final_score'])
                for scores in doc_scores.values()
                if scores['final_score'] >= self.config.hybrid_score_threshold
            ]
            
            final_results.sort(key=lambda x: x[1], reverse=True)
            return final_results[:self.config.k]
            
        except Exception as e:
            logger.error(f"Error in weighted fusion: {e}")
            return vector_results[:self.config.k]

    def search(self, query: str, vector_store: PGVector) -> List[Document]:
        """Perform hybrid search"""
        try:
            # Validate query
            if not query or not query.strip():
                logger.warning("Empty query provided for search")
                return []
            
            # Initialize TF-IDF if needed for hybrid/keyword search
            if self.config.search_type in [SearchType.HYBRID, SearchType.HYBRID_RRF, SearchType.KEYWORD_ONLY]:
                if not self.tfidf_vectorizer:
                    if not self._initialize_tfidf_vectorizer():
                        logger.warning("TF-IDF not available, falling back to vector search only")
                        # Temporarily change search type for this query
                        original_search_type = self.config.search_type
                        self.config.search_type = SearchType.VECTOR_ONLY
                        vector_results = self._vector_search(query, vector_store, self.config.k)
                        self.config.search_type = original_search_type
                        return [doc for doc, _ in vector_results]
            
            # Pure vector search
            if self.config.search_type == SearchType.VECTOR_ONLY:
                vector_results = self._vector_search(query, vector_store, self.config.k)
                return [doc for doc, _ in vector_results]
            
            # Pure keyword search
            elif self.config.search_type == SearchType.KEYWORD_ONLY:
                keyword_results = self._keyword_search(query, self.config.k)
                if not keyword_results:
                    logger.warning("Keyword search failed, falling back to vector search")
                    vector_results = self._vector_search(query, vector_store, self.config.k)
                    return [doc for doc, _ in vector_results]
                
                documents = []
                for doc_idx, _ in keyword_results:
                    if doc_idx < len(self.document_texts):
                        doc_text = self.document_texts[doc_idx]
                        doc_metadata = self.document_metadata[doc_idx] if doc_idx < len(self.document_metadata) else {}
                        documents.append(Document(page_content=doc_text, metadata=doc_metadata))
                return documents
            
            # Hybrid search
            else:
                # Get results from both searches
                vector_results = self._vector_search(query, vector_store, self.config.vector_k)
                keyword_results = self._keyword_search(query, self.config.keyword_k)
                
                # If keyword search failed, fall back to vector only
                if not keyword_results and vector_results:
                    logger.warning("Keyword search failed in hybrid mode, using vector results only")
                    return [doc for doc, _ in vector_results[:self.config.k]]
                
                # If vector search failed, fall back to keyword only  
                if not vector_results and keyword_results:
                    logger.warning("Vector search failed in hybrid mode, using keyword results only")
                    documents = []
                    for doc_idx, _ in keyword_results[:self.config.k]:
                        if doc_idx < len(self.document_texts):
                            doc_text = self.document_texts[doc_idx]
                            doc_metadata = self.document_metadata[doc_idx] if doc_idx < len(self.document_metadata) else {}
                            documents.append(Document(page_content=doc_text, metadata=doc_metadata))
                    return documents
                
                # Both searches successful, combine results
                if vector_results or keyword_results:
                    if self.config.search_type == SearchType.HYBRID_RRF:
                        combined_results = self._apply_reciprocal_rank_fusion(vector_results, keyword_results)
                    else:
                        combined_results = self._apply_weighted_fusion(vector_results, keyword_results)
                    
                    return [doc for doc, _ in combined_results]
                
                # Both searches failed
                logger.error("Both vector and keyword searches failed")
                return []
                
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to vector search
            try:
                vector_results = self._vector_search(query, vector_store, self.config.k)
                return [doc for doc, _ in vector_results]
            except Exception as fallback_error:
                logger.error(f"Fallback vector search also failed: {fallback_error}")
                return []

    def invalidate_cache(self):
        """Invalidate cached vectorizer and texts"""
        try:
            if os.path.exists(self.tfidf_cache_path):
                os.remove(self.tfidf_cache_path)
            if os.path.exists(self.texts_cache_path):
                os.remove(self.texts_cache_path)
            
            self.tfidf_vectorizer = None
            self.document_texts = []
            self.document_metadata = []
            
            logger.info("Hybrid search cache invalidated")
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")

    def debug_status(self) -> Dict[str, Any]:
        """Get detailed debug information about hybrid search status"""
        try:
            debug_info = {
                'config': {
                    'search_type': self.config.search_type.value,
                    'vector_weight': self.config.vector_weight,
                    'keyword_weight': self.config.keyword_weight,
                    'k': self.config.k,
                    'tfidf_max_features': self.config.tfidf_max_features,
                    'tfidf_min_df': self.config.tfidf_min_df,
                    'tfidf_max_df': self.config.tfidf_max_df
                },
                'cache_status': {
                    'cache_dir_exists': os.path.exists(self.cache_dir),
                    'tfidf_cache_exists': os.path.exists(self.tfidf_cache_path),
                    'texts_cache_exists': os.path.exists(self.texts_cache_path)
                },
                'data_status': {
                    'document_texts_count': len(self.document_texts),
                    'document_metadata_count': len(self.document_metadata),
                    'tfidf_vectorizer_loaded': self.tfidf_vectorizer is not None,
                    'tfidf_vocabulary_size': len(self.tfidf_vectorizer.vocabulary_) if self.tfidf_vectorizer else 0
                },
                'database_status': {}
            }
            
            # Check database connection and collection
            try:
                with self._get_database_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        # Check collection exists
                        cur.execute("""
                            SELECT uuid, name FROM langchain_pg_collection WHERE name = %s
                        """, (self.collection_name,))
                        collection = cur.fetchone()
                        
                        if collection:
                            debug_info['database_status']['collection_found'] = True
                            debug_info['database_status']['collection_uuid'] = str(collection['uuid'])
                            
                            # Count documents in collection
                            cur.execute("""
                                SELECT COUNT(*) as doc_count
                                FROM langchain_pg_embedding e
                                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                                WHERE c.name = %s
                            """, (self.collection_name,))
                            count_result = cur.fetchone()
                            debug_info['database_status']['document_count'] = count_result['doc_count'] if count_result else 0
                        else:
                            debug_info['database_status']['collection_found'] = False
                            debug_info['database_status']['document_count'] = 0
                            
            except Exception as db_error:
                debug_info['database_status']['error'] = str(db_error)
            
            # Sample some document texts if available
            if self.document_texts:
                debug_info['data_status']['sample_texts'] = [
                    text[:100] + "..." if len(text) > 100 else text
                    for text in self.document_texts[:3]
                ]
                debug_info['data_status']['empty_texts_count'] = sum(1 for text in self.document_texts if not text or not text.strip())
            
            return debug_info
            
        except Exception as e:
            return {'error': str(e)}

    def force_reload_from_database(self) -> bool:
        """Force reload of document texts from database, bypassing cache"""
        try:
            logger.info("Force reloading document texts from database")
            
            # Clear cache files
            if os.path.exists(self.tfidf_cache_path):
                os.remove(self.tfidf_cache_path)
                logger.info("Removed TF-IDF cache")
                
            if os.path.exists(self.texts_cache_path):
                os.remove(self.texts_cache_path)
                logger.info("Removed texts cache")
            
            # Clear in-memory data
            self.tfidf_vectorizer = None
            self.document_texts = []
            self.document_metadata = []
            
            # Reload from database
            success = self._load_document_texts()
            if success:
                logger.info(f"Successfully reloaded {len(self.document_texts)} documents")
                # Re-initialize TF-IDF
                tfidf_success = self._initialize_tfidf_vectorizer()
                logger.info(f"TF-IDF re-initialization: {'success' if tfidf_success else 'failed'}")
                return tfidf_success
            else:
                logger.error("Failed to reload documents from database")
                return False
                
        except Exception as e:
            logger.error(f"Error in force reload: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid search statistics"""
        return {
            'search_type': self.config.search_type.value,
            'vector_weight': self.config.vector_weight,
            'keyword_weight': self.config.keyword_weight,
            'k': self.config.k,
            'tfidf_vocabulary_size': len(self.tfidf_vectorizer.vocabulary_) if self.tfidf_vectorizer else 0,
            'document_count': len(self.document_texts),
            'cache_available': os.path.exists(self.tfidf_cache_path)
        }
