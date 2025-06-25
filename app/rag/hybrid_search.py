"""
Hybrid Search Configuration and Implementation
Combines dense vector search with sparse keyword search (BM25)
Based on LangChain hybrid search patterns
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
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

logger = logging.getLogger(__name__)


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
    vector_weight: float = 0.64
    keyword_weight: float = 0.36 
    
    # Result configuration
    k: int = 5
    keyword_k: int = 16
    vector_k: int = 16
    
    # Scoring thresholds
    vector_score_threshold: float = 0.32
    keyword_score_threshold: float = 0.16
    hybrid_score_threshold: float = 0.24
    
    # RRF parameters
    rrf_k: int = 60

    # TF-IDF parameters
    tfidf_max_features: int = 10000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_min_df: int = 3
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
                with open(self.texts_cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.document_texts = cache_data.get('texts', [])
                    self.document_metadata = cache_data.get('metadata', [])
                    
                logger.info(f"Loaded {len(self.document_texts)} document texts from cache")
                return len(self.document_texts) > 0
            
            # Load from database
            with self._get_database_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT e.document, e.cmetadata
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %s
                        ORDER BY e.id
                    """, (self.collection_name,))
                    
                    results = cur.fetchall()
                    
                    self.document_texts = []
                    self.document_metadata = []
                    
                    for row in results:
                        self.document_texts.append(row['document'])
                        self.document_metadata.append(row['cmetadata'] or {})
            
            # Cache the results
            cache_data = {
                'texts': self.document_texts,
                'metadata': self.document_metadata
            }
            with open(self.texts_cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Loaded and cached {len(self.document_texts)} document texts")
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
            
            # Initialize and fit vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                ngram_range=self.config.tfidf_ngram_range,
                min_df=self.config.tfidf_min_df,
                max_df=self.config.tfidf_max_df,
                stop_words=None,  # Keep all words for multilingual support
                lowercase=True
            )
            
            # Fit the vectorizer
            self.tfidf_vectorizer.fit(self.document_texts)
            
            # Cache the vectorizer
            with open(self.tfidf_cache_path, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            logger.info(f"Trained and cached TF-IDF vectorizer with {len(self.tfidf_vectorizer.vocabulary_)} features")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing TF-IDF vectorizer: {e}")
            return False

    def _keyword_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Perform keyword search using TF-IDF"""
        try:
            if not self.tfidf_vectorizer:
                if not self._initialize_tfidf_vectorizer():
                    return []
            
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Transform all documents
            doc_vectors = self.tfidf_vectorizer.transform(self.document_texts)
            
            # Calculate cosine similarity
            similarities = (doc_vectors * query_vector.T).toarray().flatten()
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            # Filter by threshold and return with scores
            results = []
            for idx in top_indices:
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
            # Initialize TF-IDF if needed
            if not self.tfidf_vectorizer:
                if not self._initialize_tfidf_vectorizer():
                    logger.warning("TF-IDF not available, falling back to vector search only")
                    self.config.search_type = SearchType.VECTOR_ONLY
            
            # Pure vector search
            if self.config.search_type == SearchType.VECTOR_ONLY:
                vector_results = self._vector_search(query, vector_store, self.config.k)
                return [doc for doc, _ in vector_results]
            
            # Pure keyword search
            elif self.config.search_type == SearchType.KEYWORD_ONLY:
                keyword_results = self._keyword_search(query, self.config.k)
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
                
                # Combine results
                if self.config.search_type == SearchType.HYBRID_RRF:
                    combined_results = self._apply_reciprocal_rank_fusion(vector_results, keyword_results)
                else:
                    combined_results = self._apply_weighted_fusion(vector_results, keyword_results)
                
                return [doc for doc, _ in combined_results]
                
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to vector search
            try:
                vector_results = self._vector_search(query, vector_store, self.config.k)
                return [doc for doc, _ in vector_results]
            except:
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
