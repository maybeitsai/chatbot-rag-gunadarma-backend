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


async def main():
    """Main test function"""
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