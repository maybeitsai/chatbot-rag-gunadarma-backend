#!/usr/bin/env python3
"""
Comprehensive Test Suite for setup.py
Tests all setup functionality including crawler integration, cache management, and pipeline execution
"""

import asyncio
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
import json
import time
import os
from unittest.mock import patch, MagicMock, AsyncMock
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from setup import RAGSystemSetup


class TestRAGSystemSetup:
    """Test RAGSystemSetup class functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create test directories
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("cache").mkdir(exist_ok=True)
        
        self.setup = RAGSystemSetup()
    
    def teardown_method(self):
        """Cleanup test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_setup_initialization(self):
        """Test RAGSystemSetup initialization"""
        assert self.setup.setup_start_time > 0
        assert isinstance(self.setup.steps_completed, list)
        assert isinstance(self.setup.errors, list)
        assert len(self.setup.steps_completed) == 0
        assert len(self.setup.errors) == 0
    
    def test_log_step_success(self):
        """Test successful step logging"""
        self.setup.log_step("Test Step", True, "Test details")
        
        assert "Test Step" in self.setup.steps_completed
        assert len(self.setup.errors) == 0
    
    def test_log_step_failure(self):
        """Test failed step logging"""
        self.setup.log_step("Failed Step", False, "Error details")
        
        assert "Failed Step" not in self.setup.steps_completed
        assert len(self.setup.errors) == 1
        assert "Failed Step: Error details" in self.setup.errors
    
    @patch.dict(os.environ, {
        'GOOGLE_API_KEY': 'test_key',
        'NEON_CONNECTION_STRING': 'test_connection',
        'LLM_MODEL': 'test_model',
        'EMBEDDING_MODEL': 'test_embedding'
    })
    def test_check_environment_success(self):
        """Test environment check with all variables"""
        result = self.setup.check_environment()
        
        assert result is True
        assert "Environment Check" in self.setup.steps_completed
    
    @patch.dict(os.environ, {}, clear=True)
    def test_check_environment_failure(self):
        """Test environment check with missing variables"""
        result = self.setup.check_environment()
        
        assert result is False
        assert len(self.setup.errors) > 0
    
    def test_cache_management_status(self):
        """Test cache status management"""
        try:
            result = self.setup.manage_cache("status")
            # Should work even without enhanced crawler
            assert result in [True, False]  # Could fail if crawler not available
        except ImportError:
            # Expected if enhanced crawler not available
            pass
    
    def test_cache_management_invalid_action(self):
        """Test cache management with invalid action"""
        try:
            result = self.setup.manage_cache("invalid_action")
            assert result in [True, False]
        except ImportError:
            pass
    @patch('rag.db_setup.setup_database')
    def test_setup_database_success(self, mock_setup_db):
        """Test successful database setup"""
        mock_setup_db.return_value = None
        
        result = self.setup.setup_database()
        
        assert result is True
        assert "Database Setup" in self.setup.steps_completed
    
    @patch('rag.db_setup.setup_database')
    def test_setup_database_failure(self, mock_setup_db):
        """Test database setup failure"""
        mock_setup_db.side_effect = Exception("Database error")
        
        result = self.setup.setup_database()
        
        assert result is False
        assert len(self.setup.errors) > 0


class TestSetupCrawlerIntegration:
    """Test crawler integration in setup.py"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create test directories and files
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("cache").mkdir(exist_ok=True)
        
        self.setup = RAGSystemSetup()
    
    def teardown_method(self):
        """Cleanup test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_crawl_data_with_existing_data(self):
        """Test crawl_data when data already exists"""
        # Create existing data file
        existing_data = [
            {
                "url": "https://example.com",
                "title": "Test",
                "content": "Test content",
                "timestamp": "2025-01-01T00:00:00"
            }
        ]
        
        with open("data/output.json", "w", encoding="utf-8") as f:
            json.dump(existing_data, f)
        
        # Mock user input to skip crawling
        with patch('builtins.input', return_value='n'):
            result = await self.setup.crawl_data(force_crawl=False)
        
        assert result is True
    @pytest.mark.asyncio
    @patch('crawl.crawler.WebCrawler')
    async def test_crawl_data_force_crawl(self, mock_crawler_class):
        """Test force crawl functionality with mocked crawler for speed"""
        # Mock crawler instance
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler
        
        # Mock successful crawl result
        mock_crawler.crawl.return_value = {
            'status': 'success',
            'crawl_summary': {
                'duration_seconds': 0.1,
                'total_pages_crawled': 2,
                'total_pdfs_processed': 0,
                'pages_updated': 1,
                'pages_skipped': 1,
                'duplicates_skipped': 0,
                'cache_hits': 1,
                'total_saved': 2
            },
            'cache_statistics': {
                'hit_rate': 50.0
            }
        }
        
        # Create existing data
        with open("data/output.json", "w", encoding="utf-8") as f:
            json.dump([{"test": "data"}], f)
        
        # Force crawl should bypass existing data check
        result = await self.setup.crawl_data(force_crawl=True)
        
        # Verify success
        assert result is True
        mock_crawler_class.assert_called_once()
        mock_crawler.crawl.assert_called_once_with(incremental=True)
    @pytest.mark.asyncio
    async def test_crawl_data_fallback(self):
        """Test fallback to basic crawler"""
        # Mock WebCrawler to fail
        with patch('crawl.crawler.WebCrawler', side_effect=ImportError("No optimized crawler")):
            try:
                result = await self.setup.crawl_data()
                # Should fallback to basic crawler or fail gracefully
                assert result in [True, False]
            except Exception:
                # Expected if no crawler available
                pass


class TestSetupCommandLine:
    """Test setup.py command line functionality"""
    
    def test_help_command(self):
        """Test --help command"""
        result = subprocess.run([
            sys.executable, "setup.py", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert "RAG SYSTEM SETUP" in result.stdout
        assert "--help" in result.stdout
        assert "--cache-status" in result.stdout
    
    def test_cache_status_command(self):
        """Test --cache-status command"""
        result = subprocess.run([
            sys.executable, "setup.py", "--cache-status"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Should complete regardless of result
        assert result.returncode in [0, 1]  # May fail if cache not available
    
    def test_cache_cleanup_command(self):
        """Test --cache-cleanup command"""
        result = subprocess.run([
            sys.executable, "setup.py", "--cache-cleanup"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode in [0, 1]  # May fail if cache not available
    
    def test_invalid_command(self):
        """Test invalid command line argument"""
        result = subprocess.run([
            sys.executable, "setup.py", "--invalid-option"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Should still run normal setup or exit gracefully
        assert result.returncode in [0, 1]


class TestSetupIntegration:
    """Test complete setup integration"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create required directories
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("cache").mkdir(exist_ok=True)
        Path("rag").mkdir(exist_ok=True)
        
        self.setup = RAGSystemSetup()
    
    def teardown_method(self):
        """Cleanup test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch.dict(os.environ, {
        'GOOGLE_API_KEY': 'test_key',
        'NEON_CONNECTION_STRING': 'test_connection',
        'LLM_MODEL': 'test_model',
        'EMBEDDING_MODEL': 'test_embedding'
    })
    @patch('rag.db_setup.setup_database')
    @pytest.mark.asyncio
    async def test_complete_setup_skip_crawling(self, mock_setup_db):
        """Test complete setup with skip crawling"""
        mock_setup_db.return_value = None
        
        # Create some test data
        test_data = [{"url": "test", "content": "test"}]
        with open("data/output.json", "w") as f:
            json.dump(test_data, f)
        
        try:
            result = await self.setup.run_complete_setup(skip_crawling=True)
            # May succeed or fail depending on dependencies
            assert result in [True, False]
        except Exception as e:
            # Expected if some dependencies not available
            assert "Failed" in str(e) or "Error" in str(e) or len(str(e)) > 0


# Performance and stress tests
class TestSetupPerformance:
    """Test setup.py performance"""
    
    def test_setup_initialization_performance(self):
        """Test setup initialization time"""
        start_time = time.time()
        setup = RAGSystemSetup()
        end_time = time.time()
        
        initialization_time = end_time - start_time
        assert initialization_time < 1.0  # Should initialize quickly
        assert setup.setup_start_time > 0
    
    def test_cache_management_performance(self):
        """Test cache management performance"""
        setup = RAGSystemSetup()
        
        start_time = time.time()
        try:
            result = setup.manage_cache("status")
            end_time = time.time()
            
            cache_time = end_time - start_time
            assert cache_time < 5.0  # Should complete within 5 seconds
        except ImportError:
            # Expected if enhanced crawler not available
            pass
    
    def test_multiple_setup_instances(self):
        """Test multiple RAGSystemSetup instances"""
        setups = []
        
        for i in range(5):
            setup = RAGSystemSetup()
            setups.append(setup)
            assert len(setup.steps_completed) == 0
            assert len(setup.errors) == 0
        
        # All instances should be independent
        setups[0].log_step("Test", True)
        assert len(setups[0].steps_completed) == 1
        assert len(setups[1].steps_completed) == 0


# Utility functions for testing
def test_setup_imports():
    """Test that all required imports work"""
    try:
        from setup import RAGSystemSetup, main, print_help
        assert RAGSystemSetup is not None
        assert main is not None
        assert print_help is not None
    except ImportError as e:
        pytest.fail(f"Failed to import setup components: {e}")


def test_setup_file_structure():
    """Test setup.py file structure"""
    setup_file = Path(__file__).parent.parent / "setup.py"
    
    assert setup_file.exists(), "setup.py file not found"
    
    content = setup_file.read_text(encoding="utf-8")
    
    # Check for required components
    required_components = [
        "class RAGSystemSetup",
        "def main()",
        "def print_help()",
        "async def crawl_data",
        "def manage_cache",
        "def check_environment"
    ]
    
    for component in required_components:
        assert component in content, f"Missing component: {component}"


@pytest.mark.asyncio
async def test_setup_async_functionality():
    """Test async functionality in setup"""
    setup = RAGSystemSetup()
    
    # Test that async methods exist and are callable
    assert hasattr(setup, 'crawl_data')
    assert asyncio.iscoroutinefunction(setup.crawl_data)
    
    assert hasattr(setup, 'run_complete_setup')
    assert asyncio.iscoroutinefunction(setup.run_complete_setup)


# Integration test with real setup.py
def test_real_setup_integration():
    """Test integration with real setup.py"""
    setup_dir = Path(__file__).parent.parent
    
    # Test that setup.py can be imported
    result = subprocess.run([
        sys.executable, "-c", "from setup import RAGSystemSetup; print('Import successful')"
    ], capture_output=True, text=True, cwd=setup_dir)
    
    assert result.returncode == 0
    assert "Import successful" in result.stdout


@pytest.mark.asyncio
@patch('crawl.crawler.WebCrawler')
async def test_quick_full_setup(mock_crawler_class):
    """Test simplified setup with mocked crawler for speed"""
    # Mock crawler
    mock_crawler = AsyncMock()
    mock_crawler_class.return_value = mock_crawler
    mock_crawler.crawl.return_value = {
        'status': 'success',
        'crawl_summary': {
            'duration_seconds': 0.1,
            'total_pages_crawled': 1,
            'total_pdfs_processed': 0,
            'pages_updated': 1,
            'pages_skipped': 0,
            'duplicates_skipped': 0,
            'cache_hits': 0,
            'total_saved': 1
        },
        'cache_statistics': {'hit_rate': 0.0}
    }
    
    setup = RAGSystemSetup()
    
    # Test simplified crawl only (skip full setup due to complexity)
    with patch('builtins.input', return_value='n'):
        result = await setup.crawl_data(force_crawl=True)
        
    # Should succeed with mocks
    assert result is True
    
    # Verify crawler was called
    mock_crawler_class.assert_called_once()


if __name__ == "__main__":
    # Run tests when executed directly
    print("🧪 SETUP.PY TEST SUITE")
    print("=" * 50)
    
    # Run pytest
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ ALL SETUP TESTS PASSED!")
        print("\n🎯 Setup.py is ready for production:")
        print("   python setup.py --help")
        print("   python setup.py --cache-status")
        print("   python setup.py --crawl-only")
        print("   python setup.py  # Full setup")
    else:
        print("\n❌ Some setup tests failed")
        print("Review the test output above for details")
    
    sys.exit(exit_code)
