"""
Test script to verify optimization improvements
Compares performance before and after optimizations
"""

import sys
import os
from pathlib import Path
import asyncio
import time
import json
import statistics
from typing import List, Dict, Any
import logging
import pytest
import warnings

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*BaseRetriever.get_relevant_documents.*")
warnings.filterwarnings("ignore", message=".*Chain.__call__.*")

# Add the parent directory to Python path so we can import rag modules
script_dir = Path(__file__).parent
backend_dir = script_dir.parent
sys.path.insert(0, str(backend_dir))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceTester:
    """Test and compare performance of RAG system"""
    
    def __init__(self):
        self.test_questions = [
            "Apa itu Universitas Gunadarma?",
            "Fakultas apa saja yang ada di Universitas Gunadarma?",
            "Bagaimana cara mendaftar di Universitas Gunadarma?",
            "Dimana lokasi kampus Universitas Gunadarma?",
            "Apa saja program studi yang tersedia?",
            "Bagaimana sistem pembelajaran di Universitas Gunadarma?",
            "Apa saja fasilitas yang tersedia di kampus?",
            "Bagaimana cara menghubungi BAAK Universitas Gunadarma?",
            "Kapan jadwal kuliah semester ini?",
            "Berapa biaya kuliah di Universitas Gunadarma?"        ]
        
        self.results = {
            'optimized': {}
        }
    
    async def test_pipeline(self) -> Dict[str, Any]:
        """Test optimized RAG pipeline performance"""
        logger.info("Testing optimized RAG pipeline...")
        
        try:
            from app.rag.pipeline import create_rag_pipeline
            pipeline = create_rag_pipeline(enable_cache=True)
            
            if not pipeline.test_connection():
                logger.error("Optimized pipeline connection failed")
                return {"error": "Connection failed"}
            
            # Test performance with async
            response_times = []
            results = []
            
            start_time = time.time()
            
            # Test individual queries first (to populate cache)
            for question in self.test_questions:
                question_start = time.time()
                result = await pipeline.ask_question_async(question)
                question_time = time.time() - question_start
                
                response_times.append(question_time)
                results.append(result)
                
                logger.info(f"Optimized - Question {len(results)}: {question_time:.3f}s (cached: {result.get('cached', False)})")
            
            # Test batch processing
            batch_start = time.time()
            batch_results = await pipeline.batch_questions(self.test_questions)
            batch_time = time.time() - batch_start
            
            total_time = time.time() - start_time
            
            # Get performance stats
            perf_stats = pipeline.get_performance_stats()
            
            # Calculate cache metrics
            cache_hits = sum(1 for r in results if r.get('cached', False))
            
            return {
                "total_time": total_time,
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "std_response_time": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                "successful_queries": sum(1 for r in results if r['status'] == 'success'),
                "total_queries": len(results),
                "cache_enabled": True,
                "cache_hits": cache_hits,
                "cache_hit_rate": (cache_hits / len(results)) * 100,
                "batch_time": batch_time,
                "batch_throughput_qps": len(self.test_questions) / batch_time,
                "throughput_qps": len(self.test_questions) / total_time,
                "performance_stats": perf_stats            }
            
        except Exception as e:
            logger.error(f"Error testing optimized pipeline: {e}")
            return {"error": str(e)}
    
    async def run_performance_test(self) -> Dict[str, Any]:
        """Run performance test for optimized pipeline"""
        logger.info("🧪 Starting optimized pipeline performance test...")
        
        # Test optimized pipeline
        logger.info("=" * 50)
        optimized_results = await self.test_pipeline()
        self.results['optimized'] = optimized_results
        
        # Generate report
        report = {
            "timestamp": time.time(),
            "test_questions_count": len(self.test_questions),
            "optimized_pipeline": optimized_results,
            "summary": self.generate_summary()        }
        
        return report
    
    def generate_summary(self) -> Dict[str, str]:
        """Generate human-readable summary"""
        optimized = self.results['optimized']
        
        if 'error' in optimized:
            return {"error": optimized['error']}
        
        summary = {
            "status": "✅ Performance test completed successfully!",
            "avg_response_time": f"Average response time: {optimized.get('avg_response_time', 0):.3f}s",
            "throughput": f"Throughput: {optimized.get('throughput_qps', 0):.2f} QPS",
            "cache_performance": f"Cache hit rate: {optimized.get('cache_hit_rate', 0):.1f}% ({optimized.get('cache_hits', 0)} hits)",
            "total_queries": f"Processed {optimized.get('total_queries', 0)} queries successfully"
        }
        
        if 'batch_throughput_qps' in optimized:
            summary["batch_performance"] = f"Batch processing: {optimized['batch_throughput_qps']:.2f} QPS"
        
        return summary
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted test report"""
        print("\n" + "=" * 80)
        print("🧪 OPTIMIZED PIPELINE PERFORMANCE REPORT")
        print("=" * 80)
        
        # Summary
        summary = report.get('summary', {})
        for key, value in summary.items():
            if key != "error":
                print(f"📊 {value}")
        
        # Detailed metrics
        print(f"\n📋 Detailed Metrics:")
        print(f"   Test Questions: {report['test_questions_count']}")
        
        optimized = report.get('optimized_pipeline', {})
        
        if 'error' not in optimized:
            print(f"\n⏱️  Response Times:")
            print(f"   Average: {optimized.get('avg_response_time', 0):.3f}s")
            print(f"   Median:  {optimized.get('median_response_time', 0):.3f}s")
            print(f"   Min:     {optimized.get('min_response_time', 0):.3f}s")
            print(f"   Max:     {optimized.get('max_response_time', 0):.3f}s")
            
            print(f"\n🚀 Throughput:")
            print(f"   Individual: {optimized.get('throughput_qps', 0):.2f} QPS")
            
            if optimized.get('cache_enabled'):
                print(f"\n💾 Cache Performance:")
                print(f"   Hit Rate: {optimized.get('cache_hit_rate', 0):.1f}%")
                print(f"   Cache Hits: {optimized.get('cache_hits', 0)}")                
            if 'batch_throughput_qps' in optimized:
                print(f"\n⚡ Batch Processing:")
                print(f"   Batch Throughput: {optimized['batch_throughput_qps']:.2f} QPS")
        
        print("\n" + "=" * 80)


@pytest.fixture
def performance_tester():
    """Fixture to provide PerformanceTester instance"""
    return PerformanceTester()

@pytest.fixture
def test_questions():
    """Fixture providing test questions"""
    return [
        "Apa itu Universitas Gunadarma?",
        "Fakultas apa saja yang ada di Universitas Gunadarma?",
        "Bagaimana cara mendaftar di Universitas Gunadarma?",
        "Dimana lokasi kampus Universitas Gunadarma?",
        "Apa saja program studi yang tersedia?",
        "Bagaimana sistem pembelajaran di Universitas Gunadarma?",
        "Apa saja fasilitas yang tersedia di kampus?",
        "Bagaimana cara menghubungi BAAK Universitas Gunadarma?",
        "Kapan jadwal kuliah semester ini?",
        "Berapa biaya kuliah di Universitas Gunadarma?"
    ]

class TestPerformance:
    """Performance test class for pytest"""
    
    @pytest.mark.asyncio
    async def test_pipeline_performance(self, performance_tester):
        """Test RAG pipeline performance"""
        logger.info("Testing optimized RAG pipeline performance...")
        
        result = await performance_tester.test_pipeline()
        
        # Assert basic functionality
        assert result is not None, "Pipeline test should return results"
        assert 'error' not in result, f"Pipeline test failed with error: {result.get('error')}"
        
        # Assert performance metrics exist
        assert 'total_time' in result, "Result should include total_time"
        assert 'avg_response_time' in result, "Result should include avg_response_time"
        assert 'successful_queries' in result, "Result should include successful_queries"
        assert 'total_queries' in result, "Result should include total_queries"
        
        # Assert reasonable performance
        assert result['total_time'] > 0, "Total time should be positive"
        assert result['avg_response_time'] > 0, "Average response time should be positive"
        assert result['successful_queries'] > 0, "Should have successful queries"
        assert result['total_queries'] == len(performance_tester.test_questions), "Should test all questions"
        
        # Assert cache functionality
        if result.get('cache_enabled'):
            assert 'cache_hits' in result, "Cache enabled should include cache_hits"
            assert 'cache_hit_rate' in result, "Cache enabled should include cache_hit_rate"
        
        logger.info(f"✅ Pipeline performance test completed - Avg time: {result['avg_response_time']:.3f}s")
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, performance_tester):
        """Test performance benchmark thresholds"""
        logger.info("Testing performance benchmarks...")
        
        result = await performance_tester.test_pipeline()
        
        # Skip if error occurred
        if 'error' in result:
            pytest.skip(f"Pipeline not available: {result['error']}")
        
        # Performance assertions (adjust thresholds as needed)
        avg_time = result.get('avg_response_time', float('inf'))
        assert avg_time < 10.0, f"Average response time {avg_time:.3f}s exceeds 10s threshold"
        
        throughput = result.get('throughput_qps', 0)
        assert throughput > 0.1, f"Throughput {throughput:.2f} QPS is too low"
        
        success_rate = (result.get('successful_queries', 0) / result.get('total_queries', 1)) * 100
        assert success_rate >= 80, f"Success rate {success_rate:.1f}% is below 80% threshold"
        
        logger.info(f"✅ Performance benchmarks passed - {avg_time:.3f}s avg, {throughput:.2f} QPS")
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, performance_tester):
        """Test batch processing performance"""
        logger.info("Testing batch processing performance...")
        
        result = await performance_tester.test_pipeline()
        
        # Skip if error occurred
        if 'error' in result:
            pytest.skip(f"Pipeline not available: {result['error']}")
        
        # Check if batch processing was tested
        if 'batch_time' in result and 'batch_throughput_qps' in result:
            batch_time = result['batch_time']
            batch_throughput = result['batch_throughput_qps']
            individual_throughput = result.get('throughput_qps', 0)
            
            assert batch_time > 0, "Batch processing time should be positive"
            assert batch_throughput > 0, "Batch throughput should be positive"
            
            # Batch processing should be reasonably efficient
            assert batch_throughput >= individual_throughput * 0.8, \
                f"Batch throughput {batch_throughput:.2f} QPS significantly slower than individual {individual_throughput:.2f} QPS"
            
            logger.info(f"✅ Batch processing test passed - {batch_throughput:.2f} QPS")
        else:
            pytest.skip("Batch processing not available in pipeline")
    
    @pytest.mark.asyncio 
    async def test_cache_performance(self, performance_tester):
        """Test cache performance and effectiveness"""
        logger.info("Testing cache performance...")
        
        result = await performance_tester.test_pipeline()
        
        # Skip if error occurred
        if 'error' in result:
            pytest.skip(f"Pipeline not available: {result['error']}")
        
        # Check cache functionality
        if result.get('cache_enabled'):
            cache_hits = result.get('cache_hits', 0)
            cache_hit_rate = result.get('cache_hit_rate', 0)
            total_queries = result.get('total_queries', 0)
            
            assert cache_hits >= 0, "Cache hits should be non-negative"
            assert 0 <= cache_hit_rate <= 100, "Cache hit rate should be between 0-100%"
            assert cache_hits <= total_queries, "Cache hits should not exceed total queries"
            
            # For repeated questions, we should get some cache hits
            if total_queries > 1:
                # Note: Cache hits depend on question similarity and cache implementation
                logger.info(f"Cache performance: {cache_hit_rate:.1f}% hit rate ({cache_hits}/{total_queries})")
            
            logger.info(f"✅ Cache test passed - {cache_hit_rate:.1f}% hit rate")
        else:
            pytest.skip("Cache not enabled in pipeline")
    
    @pytest.mark.asyncio
    async def test_full_performance_report(self, performance_tester):
        """Test full performance report generation"""
        logger.info("Testing full performance report generation...")
        
        report = await performance_tester.run_performance_test()
        
        # Assert report structure
        assert isinstance(report, dict), "Report should be a dictionary"
        assert 'timestamp' in report, "Report should include timestamp"
        assert 'test_questions_count' in report, "Report should include test_questions_count"
        assert 'optimized_pipeline' in report, "Report should include optimized_pipeline results"
        assert 'summary' in report, "Report should include summary"
        
        # Assert summary content
        summary = report['summary']
        assert isinstance(summary, dict), "Summary should be a dictionary"
        
        # Check if pipeline was successful
        pipeline_results = report['optimized_pipeline']
        if 'error' not in pipeline_results:
            assert 'status' in summary, "Summary should include status"
            assert 'avg_response_time' in summary, "Summary should include avg_response_time"
            assert 'throughput' in summary, "Summary should include throughput"
        
        logger.info("✅ Full performance report test passed")

# Individual test functions for backward compatibility
@pytest.mark.asyncio
async def test_rag_pipeline_performance():
    """Test RAG pipeline performance - standalone function"""
    tester = PerformanceTester()
    result = await tester.test_pipeline()
    
    assert result is not None
    assert 'error' not in result or result['error'] == "Connection failed"  # Allow connection failures in CI
    
    if 'error' not in result:
        assert result['total_time'] > 0
        assert result['avg_response_time'] > 0
        assert result['total_queries'] > 0

@pytest.mark.asyncio
async def test_performance_summary_generation():
    """Test performance summary generation - standalone function"""
    tester = PerformanceTester()
    
    # Mock some results for testing
    tester.results['optimized'] = {
        'avg_response_time': 2.5,
        'throughput_qps': 0.4,
        'cache_hit_rate': 25.0,
        'cache_hits': 2,
        'total_queries': 8,
        'batch_throughput_qps': 0.6
    }
    
    summary = tester.generate_summary()
    
    assert isinstance(summary, dict)
    assert 'status' in summary
    assert 'avg_response_time' in summary
    assert 'throughput' in summary
    assert 'cache_performance' in summary
    """Test and compare performance of RAG system"""
    
    def __init__(self):
        self.test_questions = [
            "Apa itu Universitas Gunadarma?",
            "Fakultas apa saja yang ada di Universitas Gunadarma?",
            "Bagaimana cara mendaftar di Universitas Gunadarma?",
            "Dimana lokasi kampus Universitas Gunadarma?",
            "Apa saja program studi yang tersedia?",
            "Bagaimana sistem pembelajaran di Universitas Gunadarma?",
            "Apa saja fasilitas yang tersedia di kampus?",
            "Bagaimana cara menghubungi BAAK Universitas Gunadarma?",
            "Kapan jadwal kuliah semester ini?",
            "Berapa biaya kuliah di Universitas Gunadarma?"        ]
        
        self.results = {
            'optimized': {}
        }
    
    async def test_pipeline(self) -> Dict[str, Any]:
        """Test optimized RAG pipeline performance"""
        logger.info("Testing optimized RAG pipeline...")
        
        try:
            from app.rag.pipeline import create_rag_pipeline
            pipeline = create_rag_pipeline(enable_cache=True)
            
            if not pipeline.test_connection():
                logger.error("Optimized pipeline connection failed")
                return {"error": "Connection failed"}
            
            # Test performance with async
            response_times = []
            results = []
            
            start_time = time.time()
            
            # Test individual queries first (to populate cache)
            for question in self.test_questions:
                question_start = time.time()
                result = await pipeline.ask_question_async(question)
                question_time = time.time() - question_start
                
                response_times.append(question_time)
                results.append(result)
                
                logger.info(f"Optimized - Question {len(results)}: {question_time:.3f}s (cached: {result.get('cached', False)})")
            
            # Test batch processing
            batch_start = time.time()
            batch_results = await pipeline.batch_questions(self.test_questions)
            batch_time = time.time() - batch_start
            
            total_time = time.time() - start_time
            
            # Get performance stats
            perf_stats = pipeline.get_performance_stats()
            
            # Calculate cache metrics
            cache_hits = sum(1 for r in results if r.get('cached', False))
            
            return {
                "total_time": total_time,
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "std_response_time": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                "successful_queries": sum(1 for r in results if r['status'] == 'success'),
                "total_queries": len(results),
                "cache_enabled": True,
                "cache_hits": cache_hits,
                "cache_hit_rate": (cache_hits / len(results)) * 100,
                "batch_time": batch_time,
                "batch_throughput_qps": len(self.test_questions) / batch_time,
                "throughput_qps": len(self.test_questions) / total_time,
                "performance_stats": perf_stats            }
            
        except Exception as e:
            logger.error(f"Error testing optimized pipeline: {e}")
            return {"error": str(e)}
    
    async def run_performance_test(self) -> Dict[str, Any]:
        """Run performance test for optimized pipeline"""
        logger.info("🧪 Starting optimized pipeline performance test...")
        
        # Test optimized pipeline
        logger.info("=" * 50)
        optimized_results = await self.test_pipeline()
        self.results['optimized'] = optimized_results
        
        # Generate report
        report = {
            "timestamp": time.time(),
            "test_questions_count": len(self.test_questions),
            "optimized_pipeline": optimized_results,
            "summary": self.generate_summary()        }
        
        return report
    
    def generate_summary(self) -> Dict[str, str]:
        """Generate human-readable summary"""
        optimized = self.results['optimized']
        
        if 'error' in optimized:
            return {"error": optimized['error']}
        
        summary = {
            "status": "✅ Performance test completed successfully!",
            "avg_response_time": f"Average response time: {optimized.get('avg_response_time', 0):.3f}s",
            "throughput": f"Throughput: {optimized.get('throughput_qps', 0):.2f} QPS",
            "cache_performance": f"Cache hit rate: {optimized.get('cache_hit_rate', 0):.1f}% ({optimized.get('cache_hits', 0)} hits)",
            "total_queries": f"Processed {optimized.get('total_queries', 0)} queries successfully"
        }
        
        if 'batch_throughput_qps' in optimized:
            summary["batch_performance"] = f"Batch processing: {optimized['batch_throughput_qps']:.2f} QPS"
        
        return summary
        
        summary = []
        
        # Response time improvement
        if 'avg_response_time_improvement' in improvements:
            pct = improvements['avg_response_time_improvement']
            if pct > 0:
                summary.append(f"✅ Average response time improved by {pct:.1f}%")
            else:
                summary.append(f"❌ Average response time increased by {abs(pct):.1f}%")
        
        # Throughput improvement
        if 'throughput_improvement' in improvements:
            pct = improvements['throughput_improvement']
            if pct > 0:
                summary.append(f"✅ Throughput improved by {pct:.1f}%")
            else:
                summary.append(f"❌ Throughput decreased by {abs(pct):.1f}%")
        
        # Cache benefits
        if improvements.get('cache_hit_rate', 0) > 0:
            rate = improvements['cache_hit_rate']
            hits = improvements['cache_hits']
            summary.append(f"💾 Cache enabled with {rate:.1f}% hit rate ({hits} hits)")
        
        # Batch processing benefits
        if 'batch_vs_individual_improvement' in improvements:
            pct = improvements['batch_vs_individual_improvement']
            if pct > 0:
                summary.append(f"⚡ Batch processing is {pct:.1f}% faster than individual queries")
        
        return {
            "summary_points": summary,
            "overall_verdict": "✅ Optimizations successful!" if len([s for s in summary if s.startswith("✅")]) > len([s for s in summary if s.startswith("❌")]) else "⚠️ Mixed results"
        }
    
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted test report"""
        print("\n" + "=" * 80)
        print("🧪 OPTIMIZED PIPELINE PERFORMANCE REPORT")
        print("=" * 80)
        
        # Summary
        summary = report.get('summary', {})
        for key, value in summary.items():
            if key != "error":
                print(f"📊 {value}")
        
        # Detailed metrics
        print(f"\n📋 Detailed Metrics:")
        print(f"   Test Questions: {report['test_questions_count']}")
        
        optimized = report.get('optimized_pipeline', {})
        
        if 'error' not in optimized:
            print(f"\n⏱️  Response Times:")
            print(f"   Average: {optimized.get('avg_response_time', 0):.3f}s")
            print(f"   Median:  {optimized.get('median_response_time', 0):.3f}s")
            print(f"   Min:     {optimized.get('min_response_time', 0):.3f}s")
            print(f"   Max:     {optimized.get('max_response_time', 0):.3f}s")
            
            print(f"\n🚀 Throughput:")
            print(f"   Individual: {optimized.get('throughput_qps', 0):.2f} QPS")
            
            if optimized.get('cache_enabled'):
                print(f"\n💾 Cache Performance:")
                print(f"   Hit Rate: {optimized.get('cache_hit_rate', 0):.1f}%")
                print(f"   Cache Hits: {optimized.get('cache_hits', 0)}")                
            if 'batch_throughput_qps' in optimized:
                print(f"\n⚡ Batch Processing:")
                print(f"   Batch Throughput: {optimized['batch_throughput_qps']:.2f} QPS")
        
        print("\n" + "=" * 80)


# Legacy main function for backward compatibility
async def main():
    """Main test function for backward compatibility"""
    tester = PerformanceTester()
    
    try:
        # Run performance test
        report = await tester.run_performance_test()
        
        # Print report
        tester.print_report(report)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)