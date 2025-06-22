"""
Test suite untuk Hybrid Search implementation
Menguji semua komponen hybrid search termasuk konfigurasi, algoritma, dan integrasi
"""

import pytest
import asyncio
import os
import tempfile
import json
import pickle
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Import components to test
from app.rag.hybrid_search import (
    HybridSearchConfig, 
    HybridSearchManager, 
    SearchType
)
from app.rag.vector_store import VectorStoreManager, HybridRetriever
from app.rag.pipeline import RAGPipeline
from langchain.schema import Document


@pytest.mark.unit
class TestHybridSearchConfig:
    """Test HybridSearchConfig class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = HybridSearchConfig()
        
        assert config.search_type == SearchType.HYBRID_RRF
        assert config.vector_weight == 0.7
        assert config.keyword_weight == 0.3
        assert config.k == 5  # Fixed: actual default is 5, not 8
        assert config.vector_score_threshold == 0.4  # Fixed: actual default is 0.4
        assert config.keyword_score_threshold == 0.2  # Fixed: actual default is 0.2
        assert config.rrf_k == 60
        assert config.tfidf_max_features == 10000
        assert config.tfidf_ngram_range == (1, 2)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = HybridSearchConfig(
            search_type=SearchType.HYBRID,
            vector_weight=0.8,
            keyword_weight=0.2,
            k=10,
            rrf_k=50
        )
        
        assert config.search_type == SearchType.HYBRID
        assert config.vector_weight == 0.8
        assert config.keyword_weight == 0.2
        assert config.k == 10
        assert config.rrf_k == 50
    
    def test_config_validation_valid(self):
        """Test valid configuration validation"""
        config = HybridSearchConfig(
            vector_weight=0.6,
            keyword_weight=0.4
        )
        
        assert config.validate() == True
    
    def test_config_validation_invalid_weights(self):
        """Test invalid weight configuration"""
        with pytest.raises(ValueError, match="must equal 1.0"):
            config = HybridSearchConfig(
                vector_weight=0.8,
                keyword_weight=0.3  # Sum = 1.1
            )
            config.validate()
    
    def test_config_validation_invalid_range(self):
        """Test invalid weight range"""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            config = HybridSearchConfig(
                vector_weight=1.5,
                keyword_weight=-0.5
            )
            config.validate()


@pytest.mark.unit
class TestHybridSearchManager:
    """Test HybridSearchManager class"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings model"""
        mock = Mock()
        mock.embed_query.return_value = [0.1] * 384
        return mock
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            "Universitas Gunadarma memiliki fakultas komputer dan teknologi informasi",
            "Biaya kuliah di Gunadarma sekitar 5 juta per semester",
            "Lokasi kampus Gunadarma ada di Depok dan Jakarta",
            "Program studi yang tersedia meliputi informatika dan sistem informasi",
            "Pendaftaran mahasiswa baru dibuka setiap tahun pada bulan Mei"
        ]
    
    @pytest.fixture
    def hybrid_manager(self, mock_embeddings, temp_cache_dir):
        """Create HybridSearchManager for testing"""
        config = HybridSearchConfig()
        
        # Mock cache directory
        with patch.object(HybridSearchManager, '__init__', lambda x, *args, **kwargs: None):
            manager = HybridSearchManager.__new__(HybridSearchManager)
            manager.connection_string = "mock://connection"
            manager.collection_name = "test_collection"
            manager.embeddings = mock_embeddings
            manager.config = config
            manager.tfidf_vectorizer = None
            manager.document_texts = []
            manager.document_metadata = []
            manager.cache_dir = temp_cache_dir
            manager.tfidf_cache_path = os.path.join(temp_cache_dir, "tfidf_vectorizer.pkl")
            manager.texts_cache_path = os.path.join(temp_cache_dir, "document_texts.json")
            
            return manager
    
    def test_manager_initialization(self, mock_embeddings):
        """Test manager initialization"""
        config = HybridSearchConfig()
        
        with patch('psycopg2.connect'):
            manager = HybridSearchManager(
                connection_string="mock://connection",
                collection_name="test_collection",
                embeddings=mock_embeddings,
                config=config
            )
            
            assert manager.config == config
            assert manager.embeddings == mock_embeddings
            assert manager.collection_name == "test_collection"
    def test_load_document_texts_from_cache(self, hybrid_manager, sample_documents):
        """Test loading document texts from cache"""
        # Create cache file
        cache_data = {
            'texts': sample_documents,
            'metadata': [{'url': f'http://example.com/{i}'} for i in range(len(sample_documents))]
        }
        
        with open(hybrid_manager.texts_cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f)
        
        result = hybrid_manager._load_document_texts()
        
        assert result == True
        assert hybrid_manager.document_texts == sample_documents
        assert len(hybrid_manager.document_metadata) == len(sample_documents)
    
    @patch('psycopg2.connect')
    def test_load_document_texts_from_database(self, mock_connect, hybrid_manager, sample_documents):
        """Test loading document texts from database"""
        # Mock database connection and cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {'document': text, 'cmetadata': {'url': f'http://example.com/{i}'}}
            for i, text in enumerate(sample_documents)
        ]
        
        # Properly mock the context manager
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        mock_connect.return_value = mock_conn
        
        result = hybrid_manager._load_document_texts()
        
        assert result == True
        assert hybrid_manager.document_texts == sample_documents
        assert len(hybrid_manager.document_metadata) == len(sample_documents)
    
    def test_initialize_tfidf_vectorizer_from_cache(self, hybrid_manager, sample_documents):
        """Test initializing TF-IDF vectorizer from cache"""
        # Setup documents
        hybrid_manager.document_texts = sample_documents
        
        # Create cached vectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        vectorizer.fit(sample_documents)
        
        with open(hybrid_manager.tfidf_cache_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        result = hybrid_manager._initialize_tfidf_vectorizer()
        
        assert result == True
        assert hybrid_manager.tfidf_vectorizer is not None
        assert hasattr(hybrid_manager.tfidf_vectorizer, 'vocabulary_')
    
    def test_initialize_tfidf_vectorizer_new(self, hybrid_manager, sample_documents):
        """Test initializing new TF-IDF vectorizer"""
        hybrid_manager.document_texts = sample_documents
        
        result = hybrid_manager._initialize_tfidf_vectorizer()
        
        assert result == True
        assert hybrid_manager.tfidf_vectorizer is not None
        assert len(hybrid_manager.tfidf_vectorizer.vocabulary_) > 0
    
    def test_keyword_search(self, hybrid_manager, sample_documents):
        """Test keyword search functionality"""
        hybrid_manager.document_texts = sample_documents
        hybrid_manager._initialize_tfidf_vectorizer()
        
        # Test search
        results = hybrid_manager._keyword_search("fakultas komputer", k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        
        if results:
            doc_idx, score = results[0]
            assert isinstance(doc_idx, int)
            assert isinstance(score, float)
            assert 0 <= doc_idx < len(sample_documents)
            assert score >= 0
    
    def test_vector_search_mock(self, hybrid_manager):
        """Test vector search with mock vector store"""
        mock_vector_store = Mock()
        
        # Mock similarity search results
        mock_docs = [
            (Document(page_content="test doc 1", metadata={"url": "http://test1.com"}), 0.8),
            (Document(page_content="test doc 2", metadata={"url": "http://test2.com"}), 0.6)
        ]
        mock_vector_store.similarity_search_with_score.return_value = mock_docs
        
        results = hybrid_manager._vector_search("test query", mock_vector_store, k=5)
        
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc, score in results)
        assert all(isinstance(score, float) for doc, score in results)
    
    def test_reciprocal_rank_fusion(self, hybrid_manager, sample_documents):
        """Test Reciprocal Rank Fusion algorithm"""
        # Setup documents
        hybrid_manager.document_texts = sample_documents
        hybrid_manager.document_metadata = [{'url': f'http://test{i}.com'} for i in range(len(sample_documents))]
        
        # Mock vector results
        vector_results = [
            (Document(page_content=sample_documents[0], metadata={'url': 'http://test0.com'}), 0.9),
            (Document(page_content=sample_documents[1], metadata={'url': 'http://test1.com'}), 0.7)
        ]
        
        # Mock keyword results
        keyword_results = [(1, 0.8), (2, 0.6)]
        
        results = hybrid_manager._apply_reciprocal_rank_fusion(vector_results, keyword_results)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc, score in results)
        assert all(isinstance(score, float) for doc, score in results)
    
    def test_weighted_fusion(self, hybrid_manager, sample_documents):
        """Test weighted fusion algorithm"""
        # Setup documents
        hybrid_manager.document_texts = sample_documents
        hybrid_manager.document_metadata = [{'url': f'http://test{i}.com'} for i in range(len(sample_documents))]
        
        # Mock vector results
        vector_results = [
            (Document(page_content=sample_documents[0], metadata={'url': 'http://test0.com'}), 0.9),
            (Document(page_content=sample_documents[1], metadata={'url': 'http://test1.com'}), 0.7)
        ]
        
        # Mock keyword results
        keyword_results = [(1, 0.8), (2, 0.6)]
        
        results = hybrid_manager._apply_weighted_fusion(vector_results, keyword_results)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc, score in results)
        assert all(isinstance(score, float) for doc, score in results)
    
    def test_search_vector_only(self, hybrid_manager):
        """Test vector-only search"""
        hybrid_manager.config.search_type = SearchType.VECTOR_ONLY
        
        mock_vector_store = Mock()
        mock_vector_store.similarity_search_with_score.return_value = [
            (Document(page_content="test doc", metadata={}), 0.8)
        ]
        
        results = hybrid_manager.search("test query", mock_vector_store)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc in results)
    
    def test_search_keyword_only(self, hybrid_manager, sample_documents):
        """Test keyword-only search"""
        hybrid_manager.config.search_type = SearchType.KEYWORD_ONLY
        hybrid_manager.document_texts = sample_documents
        hybrid_manager.document_metadata = [{'url': f'http://test{i}.com'} for i in range(len(sample_documents))]
        hybrid_manager._initialize_tfidf_vectorizer()
        
        mock_vector_store = Mock()
        
        results = hybrid_manager.search("fakultas komputer", mock_vector_store)
        
        assert isinstance(results, list)
        assert all(isinstance(doc, Document) for doc in results)
    
    def test_invalidate_cache(self, hybrid_manager):
        """Test cache invalidation"""
        # Create some cache files
        with open(hybrid_manager.tfidf_cache_path, 'w') as f:
            f.write("test")
        with open(hybrid_manager.texts_cache_path, 'w') as f:
            f.write("test")
        
        # Set some data
        hybrid_manager.tfidf_vectorizer = Mock()
        hybrid_manager.document_texts = ["test"]
        hybrid_manager.document_metadata = [{}]
        
        hybrid_manager.invalidate_cache()
        
        assert not os.path.exists(hybrid_manager.tfidf_cache_path)
        assert not os.path.exists(hybrid_manager.texts_cache_path)
        assert hybrid_manager.tfidf_vectorizer is None
        assert hybrid_manager.document_texts == []
        assert hybrid_manager.document_metadata == []
    
    def test_get_stats(self, hybrid_manager):
        """Test getting statistics"""
        hybrid_manager.tfidf_vectorizer = Mock()
        hybrid_manager.tfidf_vectorizer.vocabulary_ = {'test': 0, 'vocab': 1}
        hybrid_manager.document_texts = ["doc1", "doc2"]
        
        stats = hybrid_manager.get_stats()
        
        assert isinstance(stats, dict)
        assert 'search_type' in stats
        assert 'vector_weight' in stats
        assert 'keyword_weight' in stats
        assert 'tfidf_vocabulary_size' in stats
        assert 'document_count' in stats
        assert stats['document_count'] == 2
        assert stats['tfidf_vocabulary_size'] == 2


@pytest.mark.unit
class TestHybridRetriever:
    """Test HybridRetriever class"""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store"""
        return Mock()
    
    @pytest.fixture
    def mock_hybrid_manager(self):
        """Mock hybrid manager"""
        manager = Mock()
        manager.search.return_value = [
            Document(page_content="test doc", metadata={"url": "http://test.com"})
        ]
        return manager
    
    def test_retriever_initialization(self, mock_vector_store, mock_hybrid_manager):
        """Test retriever initialization"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            hybrid_manager=mock_hybrid_manager,
            k=5
        )
        
        assert retriever.vector_store == mock_vector_store
        assert retriever.hybrid_manager == mock_hybrid_manager
        assert retriever.k == 5
    
    def test_get_relevant_documents(self, mock_vector_store, mock_hybrid_manager):
        """Test getting relevant documents"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            hybrid_manager=mock_hybrid_manager
        )
        
        results = retriever.get_relevant_documents("test query")
        
        assert isinstance(results, list)
        assert len(results) > 0
        mock_hybrid_manager.search.assert_called_once_with("test query", mock_vector_store)
    
    def test_invoke_method(self, mock_vector_store, mock_hybrid_manager):
        """Test invoke method for compatibility"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            hybrid_manager=mock_hybrid_manager
        )
        
        results = retriever.invoke("test query")
        
        assert isinstance(results, list)
        mock_hybrid_manager.search.assert_called_once_with("test query", mock_vector_store)


@pytest.mark.integration
class TestVectorStoreManagerIntegration:
    """Test VectorStoreManager integration with hybrid search"""
    
    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables"""
        with patch.dict(os.environ, {
            'NEON_CONNECTION_STRING': 'mock://connection',
            'GOOGLE_API_KEY': 'mock_api_key',
            'EMBEDDING_MODEL': 'models/test-embedding'
        }):
            yield
    
    @patch('app.rag.vector_store.GoogleGenerativeAIEmbeddings')
    @patch('app.rag.vector_store.HybridSearchManager')
    def test_vector_store_manager_with_hybrid_config(self, mock_hybrid_manager, mock_embeddings, mock_env_vars):
        """Test VectorStoreManager initialization with hybrid config"""
        config = HybridSearchConfig(search_type=SearchType.HYBRID_RRF)
        
        manager = VectorStoreManager(hybrid_config=config)
        
        assert manager.hybrid_config == config
        mock_hybrid_manager.assert_called_once()
    
    @patch('app.rag.vector_store.GoogleGenerativeAIEmbeddings')
    @patch('app.rag.vector_store.HybridSearchManager')
    @patch('app.rag.vector_store.PGVector')
    def test_get_retriever_hybrid(self, mock_pgvector, mock_hybrid_manager, mock_embeddings, mock_env_vars):
        """Test getting hybrid retriever"""
        manager = VectorStoreManager()
        
        retriever = manager.get_retriever(use_hybrid=True, k=10)
        
        assert isinstance(retriever, HybridRetriever)
    
    @patch('app.rag.vector_store.GoogleGenerativeAIEmbeddings')
    @patch('app.rag.vector_store.HybridSearchManager')
    @patch('app.rag.vector_store.PGVector')
    def test_get_retriever_non_hybrid(self, mock_pgvector, mock_hybrid_manager, mock_embeddings, mock_env_vars):
        """Test getting non-hybrid retriever"""
        manager = VectorStoreManager()
        
        # Mock vector store return
        mock_vector_store = Mock()
        mock_pgvector.return_value = mock_vector_store
        mock_vector_store.as_retriever.return_value = Mock()
        
        retriever = manager.get_retriever(use_hybrid=False, k=5)
        
        mock_vector_store.as_retriever.assert_called_once()


@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Test RAG Pipeline integration with hybrid search"""
    
    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables"""
        with patch.dict(os.environ, {
            'NEON_CONNECTION_STRING': 'mock://connection',
            'GOOGLE_API_KEY': 'mock_api_key',
            'LLM_MODEL': 'gemini-2.5-flash-preview-05-20',
            'EMBEDDING_MODEL': 'models/test-embedding'
        }):
            yield
    
    @patch('app.rag.pipeline.ChatGoogleGenerativeAI')
    @patch('app.rag.pipeline.GoogleGenerativeAIEmbeddings')
    @patch('app.rag.pipeline.VectorStoreManager')
    @patch('app.rag.pipeline.get_semantic_cache')
    def test_pipeline_initialization_with_hybrid_config(self, mock_cache, mock_vector_manager, 
                                                       mock_embeddings, mock_llm, mock_env_vars):
        """Test pipeline initialization with hybrid config"""
        config = HybridSearchConfig(search_type=SearchType.HYBRID_RRF)
        
        pipeline = RAGPipeline(hybrid_config=config)
        
        assert pipeline.hybrid_config == config
        mock_vector_manager.assert_called_with(hybrid_config=config)
    
    @patch('app.rag.pipeline.ChatGoogleGenerativeAI')
    @patch('app.rag.pipeline.GoogleGenerativeAIEmbeddings')
    @patch('app.rag.pipeline.VectorStoreManager')
    @patch('app.rag.pipeline.get_semantic_cache')
    @patch('app.rag.pipeline.RetrievalQA')
    @pytest.mark.asyncio
    async def test_ask_question_async_with_hybrid(self, mock_qa, mock_cache, mock_vector_manager,
                                                  mock_embeddings, mock_llm, mock_env_vars):
        """Test async question processing with hybrid search"""
        # Setup mocks
        mock_vector_manager_instance = Mock()
        mock_vector_manager.return_value = mock_vector_manager_instance
        mock_vector_store = Mock()
        mock_vector_manager_instance.initialize_vector_store.return_value = mock_vector_store
        
        mock_cache_instance = Mock()
        mock_cache_instance.get_cached_response = AsyncMock(return_value=None)
        mock_cache_instance.cache_response = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        
        # Mock QA chain
        mock_qa_instance = Mock()
        mock_qa_result = {
            "result": "Test answer",
            "source_documents": [
                Document(page_content="doc1", metadata={"url": "http://test1.com"}),
                Document(page_content="doc2", metadata={"url": "http://test2.com"})
            ]
        }
        mock_qa_instance.return_value = mock_qa_result
        mock_qa.from_chain_type.return_value = mock_qa_instance
        
        pipeline = RAGPipeline()
        
        # Mock _get_retrieval_results to return some documents
        with patch.object(pipeline, '_get_retrieval_results', new_callable=AsyncMock) as mock_retrieval:
            mock_retrieval.return_value = [
                Document(page_content="doc1", metadata={"url": "http://test1.com"})
            ]
            
            result = await pipeline.ask_question_async(
                question="Test question",
                use_hybrid=True
            )
        
        assert isinstance(result, dict)
        assert 'answer' in result
        assert 'search_type' in result
        assert result['search_type'] == 'hybrid'


@pytest.mark.unit
class TestSearchAlgorithms:
    """Test specific search algorithms"""
    
    def test_rrf_calculation(self):
        """Test RRF score calculation"""
        # Manual RRF calculation
        k = 60
        
        # Test RRF formula: 1 / (k + rank + 1)
        rank_0_score = 1.0 / (k + 0 + 1)  # 1/61
        rank_1_score = 1.0 / (k + 1 + 1)  # 1/62
        
        assert abs(rank_0_score - (1.0/61)) < 1e-10
        assert abs(rank_1_score - (1.0/62)) < 1e-10
        assert rank_0_score > rank_1_score  # Higher rank should have higher score
    
    def test_tfidf_similarity_calculation(self):
        """Test TF-IDF similarity calculation"""
        documents = [
            "fakultas komputer universitas gunadarma",
            "biaya kuliah semester universitas",
            "program studi informatika komputer"
        ]
        
        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(documents)
        
        query = "fakultas komputer"
        query_vector = vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = (doc_vectors * query_vector.T).toarray().flatten()
        
        assert len(similarities) == len(documents)
        assert all(0 <= sim <= 1 for sim in similarities)
        assert similarities[0] > similarities[1]  # First doc should be more similar
    
    def test_weight_normalization(self):
        """Test weight normalization in weighted fusion"""
        # Test score normalization
        scores = [0.8, 0.6, 0.4, 0.2]
        
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score
        
        normalized = [(score - min_score) / score_range for score in scores]
        
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
        assert all(0 <= norm <= 1 for norm in normalized)


@pytest.mark.integration
class TestEndToEndHybridSearch:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for integration testing"""
        return [
            Document(
                page_content="Universitas Gunadarma memiliki 6 fakultas utama termasuk Fakultas Ilmu Komputer dan Teknologi Informasi",
                metadata={"url": "http://gunadarma.ac.id/fakultas", "source": "website"}
            ),
            Document(
                page_content="Biaya kuliah di Universitas Gunadarma untuk program S1 sekitar 4-6 juta rupiah per semester",
                metadata={"url": "http://gunadarma.ac.id/biaya", "source": "website"}
            ),
            Document(                page_content="Pendaftaran mahasiswa baru Universitas Gunadarma dibuka mulai bulan Maret hingga Agustus setiap tahun",
                metadata={"url": "http://gunadarma.ac.id/pendaftaran", "source": "website"}
            )
        ]
    
    @patch('psycopg2.connect')
    def test_full_hybrid_search_workflow(self, mock_connect, sample_documents):
        """Test complete hybrid search workflow"""
        # Mock database responses
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {
                'document': doc.page_content,
                'cmetadata': doc.metadata
            }
            for doc in sample_documents
        ]
        
        # Properly mock the context managers
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        mock_connect.return_value = mock_conn
        
        # Create hybrid search manager
        mock_embeddings = Mock()
        config = HybridSearchConfig(search_type=SearchType.HYBRID_RRF)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(HybridSearchManager, '__init__', lambda x, *args, **kwargs: None):
                manager = HybridSearchManager.__new__(HybridSearchManager)
                manager.connection_string = "mock://connection"
                manager.collection_name = "test_collection"
                manager.embeddings = mock_embeddings
                manager.config = config
                manager.tfidf_vectorizer = None
                manager.document_texts = []
                manager.document_metadata = []
                manager.cache_dir = temp_dir
                manager.tfidf_cache_path = os.path.join(temp_dir, "tfidf_vectorizer.pkl")
                manager.texts_cache_path = os.path.join(temp_dir, "document_texts.json")
            
                # Load documents and initialize
                manager._load_document_texts()
                manager._initialize_tfidf_vectorizer()
                
                # Mock vector store
                mock_vector_store = Mock()
                mock_vector_store.similarity_search_with_score.return_value = [
                    (sample_documents[0], 0.9),
                    (sample_documents[1], 0.7)
                ]
                
                # Perform search
                results = manager.search("fakultas komputer", mock_vector_store)
                
                assert isinstance(results, list)
                assert len(results) > 0
                assert all(isinstance(doc, Document) for doc in results)
    
    def test_config_serialization(self):
        """Test configuration serialization/deserialization"""
        config = HybridSearchConfig(
            search_type=SearchType.HYBRID_RRF,
            vector_weight=0.6,
            keyword_weight=0.4,
            k=10
        )
        
        # Serialize to dict
        config_dict = {
            'search_type': config.search_type.value,
            'vector_weight': config.vector_weight,
            'keyword_weight': config.keyword_weight,
            'k': config.k,
            'rrf_k': config.rrf_k
        }
        
        # Deserialize from dict
        new_config = HybridSearchConfig(
            search_type=SearchType(config_dict['search_type']),
            vector_weight=config_dict['vector_weight'],
            keyword_weight=config_dict['keyword_weight'],
            k=config_dict['k'],
            rrf_k=config_dict['rrf_k']
        )
        
        assert new_config.search_type == config.search_type
        assert new_config.vector_weight == config.vector_weight
        assert new_config.keyword_weight == config.keyword_weight
        assert new_config.k == config.k


if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "config":
            pytest.main(["-v", "TestHybridSearchConfig"])
        elif sys.argv[1] == "manager":
            pytest.main(["-v", "TestHybridSearchManager"])
        elif sys.argv[1] == "integration":
            pytest.main(["-v", "-m", "integration"])
        else:
            pytest.main(["-v"])
    else:
        # Run all tests
        pytest.main(["-v", "--tb=short"])
