"""
Comprehensive Test Suite for RAG System
Tests data cleaning, vector store, caching, and API endpoints
"""

import sys
import os
from pathlib import Path
import asyncio
import json
import time
import sys
from pathlib import Path
import logging
from typing import Dict, List, Any
import requests

# Note: pytest is imported only when needed for specific test functions

# Add the parent directory to Python path so we can import rag modules
script_dir = Path(__file__).parent
backend_dir = script_dir.parent
sys.path.insert(0, str(backend_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

class RAGSystemTester:
    """Comprehensive tester for the RAG system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = {
            'data_cleaning': {},
            'vector_store': {},
            'semantic_cache': {},
            'api_endpoints': {},
            'performance': {},
            'overall': {'passed': 0, 'failed': 0, 'total': 0}
        }
        
        # Test questions for validation
        self.test_questions = [
            "Apa itu Universitas Gunadarma?",
            "Fakultas apa saja yang ada di Universitas Gunadarma?", 
            "Bagaimana cara mendaftar di Universitas Gunadarma?",
            "Dimana lokasi kampus Universitas Gunadarma?",
            "Apa saja fasilitas yang tersedia di kampus?",
            "Bagaimana sistem pembelajaran di Universitas Gunadarma?",
            "Apa saja program studi yang tersedia?",
            "Bagaimana cara menghubungi BAAK Universitas Gunadarma?"
        ]
    
    def _log_test_result(self, test_name: str, passed: bool, details: Any = None):
        """Log test result and update statistics"""
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
        
        if details:
            logger.info(f"Details: {details}")
        
        self.test_results['overall']['total'] += 1
        if passed:
            self.test_results['overall']['passed'] += 1
        else:
            self.test_results['overall']['failed'] += 1
    
    async def test_data_cleaning(self):
        """Test data cleaning functionality"""
        logger.info("🧹 Testing Data Cleaning...")
        
        try:
            from app.rag.data_cleaner import DataCleaner
              # Test with sample data
            sample_data = [
                {
                    "url": "https://test.com/page1",
                    "title": "<h1>Test Title</h1>",
                    "text_content": "<p>This is test content with HTML tags that should be long enough to pass the minimum length requirement for processing. We need to make sure this content is substantial.</p>",
                    "source_type": "html"
                },
                {
                    "url": "https://test.com/page2",
                    "title": "Test Title",
                    "text_content": "This is test content with HTML tags that should be long enough to pass the minimum length requirement for processing. We need to make sure this content is substantial.",  # Similar content
                    "source_type": "html"
                },
                {
                    "url": "https://test.com/page 3",  # URL with space
                    "title": "Another Title",
                    "text_content": "Different content that should be kept and is also long enough to meet the minimum requirements for processing. This content is unique and substantial.",                    "source_type": "html"
                }
            ]
            
            cleaner = DataCleaner(similarity_threshold=0.9)
            cleaned_data = cleaner.clean_and_deduplicate_data(sample_data)
            
            # Test assertions
            assert len(cleaned_data) < len(sample_data), "Deduplication should reduce data size"
            
            # Check if URL spaces are handled (either encoded or kept as-is)
            urls_in_cleaned = [item['url'] for item in cleaned_data]
            has_space_url = any("page 3" in url or "page%203" in url for url in urls_in_cleaned)
            assert has_space_url, f"URL with spaces should be processed. Found URLs: {urls_in_cleaned}"
            
            assert not any("<" in item['title'] for item in cleaned_data), "HTML tags should be cleaned"
            
            self.test_results['data_cleaning']['sample_test'] = {
                'passed': True,
                'original_count': len(sample_data),
                'cleaned_count': len(cleaned_data),
                'reduction_rate': (len(sample_data) - len(cleaned_data)) / len(sample_data)
            }
            
            self._log_test_result("Data Cleaning - Sample Test", True, self.test_results['data_cleaning']['sample_test'])
            
        except Exception as e:
            self.test_results['data_cleaning']['sample_test'] = {'passed': False, 'error': str(e)}
            self._log_test_result("Data Cleaning - Sample Test", False, str(e))
    async def test_vector_store(self):
        """Test optimized vector store functionality"""
        logger.info("🗄️ Testing Optimized Vector Store...")
        
        try:
            from app.rag.vector_store import VectorStoreManager
            
            manager = VectorStoreManager()
            
            # Test connection and stats
            stats = manager.get_vector_store_stats()
            
            assert 'document_count' in stats, "Stats should include document count"
            assert 'collection_name' in stats, "Stats should include collection name"
            assert 'hnsw_params' in stats, "Stats should include HNSW parameters"
            
            # Test basic similarity search (without metadata filtering)
            if stats.get('document_count', 0) > 0:
                # Test retriever initialization
                retriever = manager.get_retriever(
                    search_type="similarity_score_threshold",
                    k=3,
                    score_threshold=0.3
                )
                
                assert retriever is not None, "Retriever should be initialized"
                
                # Test vector store initialization
                vector_store = manager.initialize_vector_store()
                assert vector_store is not None, "Vector store should be initialized"
                
                self.test_results['vector_store']['search_test'] = {
                    'passed': True,
                    'document_count': stats['document_count'],
                    'has_hnsw_params': True,
                    'retriever_initialized': True
                }
            else:
                self.test_results['vector_store']['search_test'] = {
                    'passed': True,
                    'document_count': 0,
                    'note': 'No documents in vector store'
                }
            
            # Test index creation capability
            try:
                manager.create_indexes()
                index_creation_status = "successful"
            except Exception as idx_error:
                index_creation_status = f"failed: {str(idx_error)}"
            
            self.test_results['vector_store']['stats'] = stats
            self.test_results['vector_store']['index_creation'] = index_creation_status
            self._log_test_result("Vector Store - Optimized Connection & Stats", True, 
                                f"Documents: {stats.get('document_count', 0)}, HNSW: {stats.get('hnsw_params', {})}")
            
        except Exception as e:
            self.test_results['vector_store']['error'] = str(e)
            self._log_test_result("Vector Store - Optimized Connection & Stats", False, str(e))
    
    async def test_semantic_cache(self):
        """Test semantic cache functionality"""
        logger.info("🧠 Testing Semantic Cache...")
        
        try:
            from app.rag.semantic_cache import SemanticCache
            
            cache = SemanticCache(
                cache_file="cache/test_cache.json",
                similarity_threshold=0.8,
                max_cache_size=10
            )
            
            # Test caching
            test_question = "Test question for caching"
            test_response = {
                "answer": "Test answer",
                "source_urls": ["https://test.com"],
                "source_count": 1,
                "status": "success"
            }
            
            # Cache response
            await cache.cache_response(test_question, test_response)
            
            # Try to retrieve
            cached_result = await cache.get_cached_response(test_question)
            
            assert cached_result is not None, "Cached response should be retrievable"
            assert cached_result['answer'] == test_response['answer'], "Cached answer should match"
            assert cached_result['cached'] == True, "Response should be marked as cached"
            
            # Test semantic similarity
            similar_question = "Test question for cache"  # Similar to above
            similar_result = await cache.get_cached_response(similar_question)
            
            cache_stats = cache.get_cache_stats()
            
            self.test_results['semantic_cache'] = {
                'exact_match': cached_result is not None,
                'semantic_match': similar_result is not None,
                'cache_stats': cache_stats
            }
            
            self._log_test_result("Semantic Cache - Functionality", True, self.test_results['semantic_cache'])
            
        except Exception as e:
            self.test_results['semantic_cache']['error'] = str(e)
            self._log_test_result("Semantic Cache - Functionality", False, str(e))
    
    def test_api_health(self):
        """Test API health endpoint"""
        logger.info("🏥 Testing API Health...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                
                self.test_results['api_endpoints']['health'] = {
                    'status_code': response.status_code,
                    'response': health_data,
                    'healthy': health_data.get('status') == 'healthy'
                }
                
                self._log_test_result("API Health Endpoint", True, health_data)
            else:
                self.test_results['api_endpoints']['health'] = {
                    'status_code': response.status_code,
                    'error': response.text
                }
                self._log_test_result("API Health Endpoint", False, f"Status: {response.status_code}")
                
        except Exception as e:
            self.test_results['api_endpoints']['health'] = {'error': str(e)}
            self._log_test_result("API Health Endpoint", False, str(e))
    
    def test_api_question(self):
        """Test API question endpoint"""
        logger.info("❓ Testing API Question Endpoint...")
        
        try:
            test_question = self.test_questions[0]
            
            payload = {
                "question": test_question,
                "use_cache": True
            }
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/ask", json=payload, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                
                assert 'answer' in response_data, "Response should contain answer"
                assert 'status' in response_data, "Response should contain status"
                assert 'source_urls' in response_data, "Response should contain source URLs"
                
                self.test_results['api_endpoints']['question'] = {
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'response': response_data,
                    'has_answer': len(response_data.get('answer', '')) > 0,
                    'status': response_data.get('status')
                }
                
                self._log_test_result("API Question Endpoint", True, {
                    'response_time': f"{response_time:.2f}s",
                    'status': response_data.get('status'),
                    'cached': response_data.get('cached', False)
                })
            else:
                self.test_results['api_endpoints']['question'] = {
                    'status_code': response.status_code,
                    'error': response.text
                }
                self._log_test_result("API Question Endpoint", False, f"Status: {response.status_code}")
                
        except Exception as e:
            self.test_results['api_endpoints']['question'] = {'error': str(e)}
            self._log_test_result("API Question Endpoint", False, str(e))
    
    def test_api_batch_questions(self):
        """Test API batch questions endpoint"""
        logger.info("📦 Testing API Batch Questions...")
        
        try:
            payload = {
                "questions": self.test_questions[:3],
                "use_cache": True
            }
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/ask/batch", json=payload, timeout=60)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                
                assert 'results' in response_data, "Response should contain results"
                assert 'total_questions' in response_data, "Response should contain total questions"
                assert len(response_data['results']) == len(payload['questions']), "Results count should match questions count"
                
                self.test_results['api_endpoints']['batch'] = {
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'total_questions': response_data.get('total_questions'),
                    'processing_time': response_data.get('processing_time'),
                    'all_answered': all(len(r.get('answer', '')) > 0 for r in response_data['results'])
                }
                
                self._log_test_result("API Batch Questions", True, {
                    'questions': len(payload['questions']),
                    'response_time': f"{response_time:.2f}s",
                    'processing_time': f"{response_data.get('processing_time', 0):.2f}s"
                })
            else:
                self.test_results['api_endpoints']['batch'] = {
                    'status_code': response.status_code,
                    'error': response.text
                }
                self._log_test_result("API Batch Questions", False, f"Status: {response.status_code}")
                
        except Exception as e:
            self.test_results['api_endpoints']['batch'] = {'error': str(e)}
            self._log_test_result("API Batch Questions", False, str(e))
    
    def test_api_stats(self):
        """Test API stats endpoint"""
        logger.info("📊 Testing API Stats Endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=10)
            
            if response.status_code == 200:
                stats_data = response.json()
                
                self.test_results['api_endpoints']['stats'] = {
                    'status_code': response.status_code,
                    'stats': stats_data
                }
                
                self._log_test_result("API Stats Endpoint", True, "Stats retrieved successfully")
            else:
                self.test_results['api_endpoints']['stats'] = {
                    'status_code': response.status_code,
                    'error': response.text
                }
                self._log_test_result("API Stats Endpoint", False, f"Status: {response.status_code}")
                
        except Exception as e:
            self.test_results['api_endpoints']['stats'] = {'error': str(e)}
            self._log_test_result("API Stats Endpoint", False, str(e))
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        logger.info("⚡ Testing Performance Benchmarks...")
        
        try:
            # Test single question performance
            question = self.test_questions[0]
            times = []
            
            for i in range(3):  # Test 3 times
                payload = {"question": question, "use_cache": True}
                
                start_time = time.time()
                response = requests.post(f"{self.base_url}/ask", json=payload, timeout=30)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    times.append(response_time)
            
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                self.test_results['performance'] = {
                    'single_question': {
                        'avg_response_time': avg_time,
                        'min_response_time': min_time,
                        'max_response_time': max_time,
                        'tests_run': len(times)
                    }
                }
                
                # Performance thresholds (adjust as needed)
                performance_good = avg_time < 5.0  # 5 seconds average
                
                self._log_test_result("Performance Benchmarks", performance_good, {
                    'avg_time': f"{avg_time:.2f}s",
                    'min_time': f"{min_time:.2f}s",
                    'max_time': f"{max_time:.2f}s"
                })
            else:
                self._log_test_result("Performance Benchmarks", False, "No successful requests")
                
        except Exception as e:
            self.test_results['performance']['error'] = str(e)
            self._log_test_result("Performance Benchmarks", False, str(e))
    
    async def run_all_tests(self, test_api: bool = True):
        """Run all tests"""
        logger.info("🚀 Starting Comprehensive RAG System Tests")
        logger.info("=" * 60)
        
        # Component tests (always run)
        await self.test_data_cleaning()
        await self.test_vector_store()
        await self.test_semantic_cache()
        
        # API tests (optional, requires running server)
        if test_api:
            logger.info("\n🌐 Testing API Endpoints (requires running server)...")
            
            # Wait a bit for server to be ready
            await asyncio.sleep(2)
            
            self.test_api_health()
            self.test_api_question()
            self.test_api_batch_questions()
            self.test_api_stats()
            await self.test_performance_benchmarks()
        
        # Generate test report
        await self.generate_test_report()
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("\n📋 Generating Test Report...")
        
        # Calculate overall success rate
        total_tests = self.test_results['overall']['total']
        passed_tests = self.test_results['overall']['passed']
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Create report
        report = {
            'timestamp': time.time(),
            'date': time.ctime(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': self.test_results['overall']['failed'],
                'success_rate': f"{success_rate:.1f}%"
            },
            'detailed_results': self.test_results
        }
        
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {self.test_results['overall']['failed']}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            logger.info("✅ OVERALL STATUS: GOOD")
        elif success_rate >= 60:
            logger.info("⚠️ OVERALL STATUS: NEEDS IMPROVEMENT")
        else:
            logger.info("❌ OVERALL STATUS: CRITICAL ISSUES")
        
        return report


async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG system components and API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--skip-api", action="store_true", help="Skip API tests")
    parser.add_argument("--components-only", action="store_true", help="Test components only")
    
    args = parser.parse_args()
    
    tester = RAGSystemTester(base_url=args.api_url)
    
    test_api = not (args.skip_api or args.components_only)
    
    try:
        await tester.run_all_tests(test_api=test_api)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Tests interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)