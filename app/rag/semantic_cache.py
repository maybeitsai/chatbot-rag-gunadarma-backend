"""
Semantic Caching System for RAG Pipeline
Reduces API calls by caching semantically similar questions and answers
"""

import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from difflib import SequenceMatcher
import logging
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached question-answer pair"""
    question: str
    question_embedding: Optional[List[float]]
    answer: str
    source_urls: List[str]
    source_count: int
    timestamp: float
    access_count: int
    last_accessed: float
    cache_key: str


class SemanticCache:
    """
    Semantic cache for RAG pipeline responses
    Uses embedding similarity to find cached answers for similar questions
    """
    
    def __init__(self, 
                 cache_file: str = "cache/semantic_cache.json",
                 similarity_threshold: float = 0.9,
                 max_cache_size: int = 1000,
                 ttl_hours: int = 24):
        
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_hours * 3600
        
        self.cache: Dict[str, CacheEntry] = {}
        self.embeddings_model = None
        
        # Load existing cache
        self._load_cache()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                for key, data in cache_data.items():
                    self.cache[key] = CacheEntry(**data)
                    
                logger.info(f"Loaded {len(self.cache)} entries from cache")
                
                # Clean expired entries
                self._cleanup_expired()
                
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            cache_data = {}
            for key, entry in self.cache.items():
                cache_data[key] = asdict(entry)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Could not save cache: {e}")
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Removed {len(expired_keys)} expired cache entries")
    
    def _cleanup_lru(self):
        """Remove least recently used entries if cache is full"""
        if len(self.cache) <= self.max_cache_size:
            return
        
        # Sort by last accessed time and remove oldest
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        entries_to_remove = len(self.cache) - self.max_cache_size
        for i in range(entries_to_remove):
            key = sorted_entries[i][0]
            del self.cache[key]
        
        logger.info(f"Removed {entries_to_remove} LRU cache entries")
    
    def _normalize_question(self, question: str) -> str:
        """Normalize question for comparison"""
        return question.lower().strip()
    
    def _generate_cache_key(self, question: str) -> str:
        """Generate cache key for question"""
        normalized = self._normalize_question(question)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using sequence matcher"""
        norm1 = self._normalize_question(text1)
        norm2 = self._normalize_question(text2)
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def _calculate_embedding_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            arr1 = np.array(emb1)
            arr2 = np.array(emb2)
            
            dot_product = np.dot(arr1, arr2)
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.warning(f"Could not calculate embedding similarity: {e}")
            return 0.0
    
    def _find_similar_cached_entry(self, question: str, question_embedding: Optional[List[float]] = None) -> Optional[CacheEntry]:
        """Find similar cached entry using text and/or embedding similarity"""
        best_match = None
        best_similarity = 0.0
        
        for entry in self.cache.values():
            # Calculate text similarity
            text_similarity = self._calculate_text_similarity(question, entry.question)
            
            # Calculate embedding similarity if available
            embedding_similarity = 0.0
            if question_embedding and entry.question_embedding:
                embedding_similarity = self._calculate_embedding_similarity(
                    question_embedding, entry.question_embedding
                )
            
            # Use the higher of the two similarities
            overall_similarity = max(text_similarity, embedding_similarity)
            
            if overall_similarity > best_similarity and overall_similarity >= self.similarity_threshold:
                best_similarity = overall_similarity
                best_match = entry
        
        if best_match:
            logger.info(f"Found similar cached entry with similarity: {best_similarity:.3f}")
        
        return best_match
    
    async def get_cached_response(self, question: str, question_embedding: Optional[List[float]] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a question
        
        Args:
            question: The question to search for
            question_embedding: Optional embedding of the question
            
        Returns:
            Cached response or None if not found
        """
        self.total_requests += 1
        
        # Clean up expired entries periodically
        if self.total_requests % 100 == 0:
            self._cleanup_expired()
        
        # Try exact match first
        cache_key = self._generate_cache_key(question)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            entry.access_count += 1
            entry.last_accessed = time.time()
            self.hits += 1
            
            logger.info(f"Cache hit (exact match) for question: {question[:50]}...")
            
            return {
                "answer": entry.answer,
                "source_urls": entry.source_urls,
                "source_count": entry.source_count,
                "status": "success",
                "cached": True,
                "cache_type": "exact"
            }
        
        # Try semantic similarity match
        similar_entry = self._find_similar_cached_entry(question, question_embedding)
        if similar_entry:
            similar_entry.access_count += 1
            similar_entry.last_accessed = time.time()
            self.hits += 1
            
            logger.info(f"Cache hit (semantic match) for question: {question[:50]}...")
            
            return {
                "answer": similar_entry.answer,
                "source_urls": similar_entry.source_urls,
                "source_count": similar_entry.source_count,
                "status": "success",
                "cached": True,
                "cache_type": "semantic"
            }
        
        self.misses += 1
        logger.info(f"Cache miss for question: {question[:50]}...")
        return None
    
    async def cache_response(self, 
                           question: str, 
                           response: Dict[str, Any], 
                           question_embedding: Optional[List[float]] = None):
        """
        Cache a question-answer pair
        
        Args:
            question: The question
            response: The response dictionary from RAG pipeline
            question_embedding: Optional embedding of the question
        """
        try:
            cache_key = self._generate_cache_key(question)
            current_time = time.time()
            
            entry = CacheEntry(
                question=question,
                question_embedding=question_embedding,
                answer=response.get("answer", ""),
                source_urls=response.get("source_urls", []),
                source_count=response.get("source_count", 0),
                timestamp=current_time,
                access_count=1,
                last_accessed=current_time,
                cache_key=cache_key
            )
            
            self.cache[cache_key] = entry
            
            # Cleanup if cache is too large
            self._cleanup_lru()
            
            # Save to disk periodically
            if len(self.cache) % 10 == 0:
                await asyncio.to_thread(self._save_cache)
            
            logger.info(f"Cached response for question: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Error caching response: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = (self.hits / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "total_entries": len(self.cache),
            "total_requests": self.total_requests,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "similarity_threshold": self.similarity_threshold,
            "max_cache_size": self.max_cache_size,
            "ttl_hours": self.ttl_seconds / 3600
        }
    
    def clear_cache(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache file: {e}")
    
    def save_cache_sync(self):
        """Synchronously save cache (for shutdown)"""
        self._save_cache()
    
    def export_cache_stats(self) -> Dict[str, Any]:
        """Export detailed cache statistics"""
        if not self.cache:
            return {"message": "Cache is empty"}
        
        entries_by_access = sorted(
            self.cache.values(),
            key=lambda x: x.access_count,
            reverse=True
        )
        
        return {
            "stats": self.get_cache_stats(),
            "top_accessed_questions": [
                {
                    "question": entry.question[:100] + "..." if len(entry.question) > 100 else entry.question,
                    "access_count": entry.access_count,
                    "last_accessed": datetime.fromtimestamp(entry.last_accessed).isoformat()
                }
                for entry in entries_by_access[:10]
            ],
            "cache_size_mb": round(len(json.dumps([asdict(e) for e in self.cache.values()])) / (1024 * 1024), 2)
        }


# Global cache instance
_semantic_cache = None

def get_semantic_cache() -> SemanticCache:
    """Get or create global semantic cache instance"""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache()
    return _semantic_cache


if __name__ == "__main__":
    # Test the semantic cache
    async def test_cache():
        cache = SemanticCache(similarity_threshold=0.8)
        
        # Test questions
        questions = [
            "Apa itu Universitas Gunadarma?",
            "Apa saja fakultas di Universitas Gunadarma?",
            "Bagaimana cara mendaftar kuliah di Gunadarma?",
            "Fakultas apa saja yang tersedia di UG?",
        ]
        
        # Simulate responses
        for i, question in enumerate(questions):
            response = {
                "answer": f"Jawaban untuk pertanyaan {i+1}: {question}",
                "source_urls": [f"https://example.com/{i+1}"],
                "source_count": 1,
                "status": "success"
            }
            
            # Try to get from cache first
            cached = await cache.get_cached_response(question)
            if cached:
                print(f"Cache hit: {question}")
                print(f"Type: {cached.get('cache_type')}")
            else:
                print(f"Cache miss: {question}")
                await cache.cache_response(question, response)
        
        # Print stats
        stats = cache.get_cache_stats()
        print(f"\nCache stats: {stats}")
    
    asyncio.run(test_cache())
