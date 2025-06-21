#!/usr/bin/env python3
"""
Unified Setup Script for RAG System
Handles complete system initialization, data processing, and optimization
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Annotated
import time
from enum import Enum
from dataclasses import dataclass
from dotenv import load_dotenv

# Rich imports for beautiful CLI
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Confirm
from rich.logging import RichHandler
from rich.tree import Tree
from rich.status import Status

# Typer for modern CLI
import typer
from typer import Typer, Option, Argument

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

# Load environment variables
load_dotenv(BACKEND_DIR / '.env')

# Initialize console and app
console = Console()
app = Typer(
    name="rag-setup",
    help="🚀 RAG System Setup - Enhanced with Advanced Caching",
    rich_markup_mode="rich",
    no_args_is_help=True
)

class SetupStep(Enum):
    """Enumeration of setup steps"""
    ENVIRONMENT = "Environment Check"
    DATABASE = "Database Setup"
    CACHE_STATUS = "Cache Status"
    DATA_CRAWL = "Data Crawling"
    DATA_PROCESS = "Data Processing"
    VECTOR_TEST = "Vector Store Test"
    SYSTEM_TEST = "System Test"
    OPTIMIZATION = "Vector Store Optimization"

@dataclass
class SetupConfig:
    """Configuration for setup process"""
    skip_crawling: bool = False
    force_crawl: bool = False
    cache_only: bool = False
    optimize_only: bool = False
    crawl_only: bool = False
    log_level: str = "INFO"

class Logger:
    """Centralized logging setup with Rich"""
    
    @staticmethod
    def setup_logging(log_level: str = "INFO") -> logging.Logger:
        """Setup logging configuration with Rich handler"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Clear existing handlers
        logging.getLogger().handlers.clear()
        
        # Setup Rich logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(console=console, rich_tracebacks=True),
                logging.FileHandler(log_dir / "setup.log", encoding='utf-8')
            ]
        )
        return logging.getLogger(__name__)

class StepTracker:
    """Track setup steps and results with Rich progress"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = time.time()
        self.completed_steps: List[str] = []
        self.errors: List[str] = []
        self.current_progress = None
    
    def log_step(self, step: SetupStep, success: bool = True, details: str = ""):
        """Log setup step completion with Rich formatting"""
        if success:
            status_icon = "✅"
            status_color = "green"
        else:
            status_icon = "❌"
            status_color = "red"
        
        step_text = Text(f"{status_icon} {step.value}", style=status_color)
        console.print(step_text)
        
        if details:
            detail_text = Text(f"   {details}", style="dim")
            console.print(detail_text)
        
        if success:
            self.completed_steps.append(step.value)
        else:
            self.errors.append(f"{step.value}: {details}")
    
    def get_duration(self) -> float:
        """Get total duration"""
        return time.time() - self.start_time
    
    def print_summary(self):
        """Print setup summary with Rich table"""
        duration = self.get_duration()
        
        table = Table(title="📊 Setup Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        table.add_row("⏱️ Duration", f"{duration:.1f} seconds")
        table.add_row("✅ Steps Completed", str(len(self.completed_steps)))
        table.add_row("❌ Errors", str(len(self.errors)))
        
        console.print()
        console.print(table)
        
        if self.errors:
            error_panel = Panel(
                "\n".join([f"• {error}" for error in self.errors]),
                title="❌ Errors Encountered",
                border_style="red"
            )
            console.print(error_panel)

class EnvironmentValidator:
    """Validate environment configuration"""
    
    REQUIRED_ENV_VARS = [
        "GOOGLE_API_KEY",
        "NEON_CONNECTION_STRING", 
        "LLM_MODEL",
        "EMBEDDING_MODEL"
    ]
    
    @classmethod
    def validate(cls) -> tuple[bool, List[str]]:
        """Validate required environment variables"""
        missing_vars = [var for var in cls.REQUIRED_ENV_VARS if not os.getenv(var)]
        return len(missing_vars) == 0, missing_vars
    
    @classmethod
    def show_env_status(cls):
        """Show environment status with Rich table"""
        table = Table(title="🔧 Environment Variables", show_header=True, header_style="bold blue")
        table.add_column("Variable", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Value Preview", style="dim")
        
        for var in cls.REQUIRED_ENV_VARS:
            value = os.getenv(var)
            if value:
                status = "✅ Set"
                preview = f"{value[:10]}..." if len(value) > 10 else value
            else:
                status = "❌ Missing"
                preview = "Not set"
            
            table.add_row(var, status, preview)
        
        console.print(table)

class CacheManager:
    """Enhanced cache management with Rich display"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def manage_cache(self, action: str) -> bool:
        """Manage crawler cache system"""
        try:
            from app.crawl.config import CrawlConfig
            from app.crawl.cache_manager import CacheManager as CrawlCacheManager
            
            config = CrawlConfig()
            cache_manager = CrawlCacheManager(config)
            
            if action == "status":
                return self._show_cache_status(cache_manager)
            elif action == "cleanup":
                return self._cleanup_cache(cache_manager)
            elif action == "clear":
                return self._clear_cache(cache_manager)
            
            return False
            
        except ImportError:
            console.print("⚠️ Enhanced cache manager not available", style="yellow")
            return False
        except Exception as e:
            console.print(f"❌ Cache management failed: {e}", style="red")
            return False
    
    def _show_cache_status(self, cache_manager) -> bool:
        """Show cache status with Rich table"""
        stats = cache_manager.get_cache_statistics()
        
        # Main stats table
        table = Table(title="📈 Cache Statistics", show_header=True, header_style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row("Total Requests", str(stats['total_requests']))
        table.add_row("Hit Rate", f"{stats['hit_rate']:.1f}%")
        table.add_row("Total Entries", str(sum(stats['cache_sizes'].values())))
        
        console.print(table)
        
        # Cache details table
        if stats['cache_sizes']:
            detail_table = Table(title="📋 Cache Details", show_header=True, header_style="bold blue")
            detail_table.add_column("Cache Type", style="cyan")
            detail_table.add_column("Entries", justify="right", style="green")
            
            for cache_type, size in stats['cache_sizes'].items():
                if size > 0:
                    detail_table.add_row(cache_type.replace('_', ' ').title(), str(size))
            
            console.print(detail_table)
        
        return True
    
    def _cleanup_cache(self, cache_manager) -> bool:
        """Cleanup expired cache with status"""
        with Status("🧹 Cleaning up expired cache entries...", console=console):
            cache_manager.cleanup_expired_cache()
        
        console.print("✅ Expired cache entries removed", style="green")
        return True
    
    def _clear_cache(self, cache_manager) -> bool:
        """Clear all cache with confirmation"""
        if not Confirm.ask("🗑️ Are you sure you want to clear ALL cache data?"):
            console.print("ℹ️ Cache clear cancelled", style="yellow")
            return False
        
        with Status("🗑️ Clearing all cache data...", console=console):
            # Clear memory cache
            for cache_attr in ['url_cache', 'content_cache', 'response_cache', 
                              'similarity_cache', 'robots_cache']:
                if hasattr(cache_manager, cache_attr):
                    getattr(cache_manager, cache_attr).clear()
            
            # Clear persistent cache files
            cache_dir = Path("cache")
            if cache_dir.exists():
                cache_files = ["url_cache.json", "response_cache.json", 
                              "similarity_cache.json", "robots_cache.json"]
                for cache_file in cache_files:
                    cache_path = cache_dir / cache_file
                    cache_path.unlink(missing_ok=True)
        
        console.print("✅ All cache data cleared", style="green")
        return True

class DataCrawler:
    """Enhanced data crawling functionality with Rich progress"""
    
    def __init__(self, logger: logging.Logger, tracker: StepTracker):
        self.logger = logger
        self.tracker = tracker
    
    async def crawl_data(self, force_crawl: bool = False) -> bool:
        """Enhanced data crawling with caching"""
        data_files = ["data/output.json", "data/output.csv"]
        existing_files = [f for f in data_files if os.path.exists(f)]
        
        if existing_files and not force_crawl:
            self.tracker.log_step(
                SetupStep.DATA_CRAWL, 
                True, 
                f"Found existing data: {', '.join(existing_files)}"
            )
            
            if not self._should_recrawl():
                return True
        
        return await self._perform_crawl()
    
    def _should_recrawl(self) -> bool:
        """Ask user if they want to recrawl with Rich prompt"""
        return Confirm.ask(
            "\n🔄 Existing data found. Recrawl with enhanced caching?",
            default=False
        )
    
    async def _perform_crawl(self) -> bool:
        """Perform the actual crawling with Rich progress"""
        console.print("🕷️ Starting enhanced crawling...", style="bold blue")
        
        try:
            from app.crawl.config import CrawlConfig
            from app.crawl.crawler import WebCrawler
            
            config = self._get_crawl_config()
            target_urls = ["https://baak.gunadarma.ac.id/", "https://www.gunadarma.ac.id/"]
            
            self._show_crawl_config(config, target_urls)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("🕷️ Crawling websites...", total=100)
                
                crawler = WebCrawler(target_urls, config)
                report = await crawler.crawl(incremental=True)
                
                progress.update(task, completed=100)
            
            return self._process_crawl_report(report)
            
        except ImportError:
            return await self._fallback_crawl()
        except Exception as e:
            console.print(f"❌ Enhanced crawling failed: {e}", style="red")
            return await self._fallback_crawl()
    
    def _get_crawl_config(self):
        """Get optimized crawl configuration"""
        from app.crawl.config import CrawlConfig
        
        return CrawlConfig(
            max_depth=0,
            similarity_threshold=0.8,
            duplicate_threshold=0.95,
            request_delay=1.0,
            baak_delay=2.0,
            enable_url_cache=True,
            enable_content_cache=True,
            enable_response_cache=True,
            enable_smart_filtering=True,
            enable_robots_respect=True,
            cache_ttl=3600,
            max_cache_size=1000,
            max_retries=3,
            timeout=60
        )
    
    def _show_crawl_config(self, config, target_urls):
        """Show crawling configuration with Rich panel"""
        config_text = f"""
🎯 Target URLs: {len(target_urls)}
💾 Caching: URL[✅] Content[✅] Response[✅]
🧠 Smart filtering: [✅]
🤖 Robots.txt respect: [✅]
🔍 Duplicate threshold: {config.duplicate_threshold}
📊 Max depth: {config.max_depth}
        """.strip()
        
        panel = Panel(
            config_text,
            title="🚀 Enhanced Crawling Configuration",
            border_style="blue"
        )
        console.print(panel)
    
    def _process_crawl_report(self, report: Dict[str, Any]) -> bool:
        """Process crawling report with Rich table"""
        if report.get('status') != 'success':
            error_msg = report.get('error', 'Unknown crawling error')
            self.tracker.log_step(SetupStep.DATA_CRAWL, False, error_msg)
            return False
        
        summary = report['crawl_summary']
        cache_stats = report.get('cache_statistics', {})
        
        # Create results table
        table = Table(title="📈 Crawling Results", show_header=True, header_style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row("⏱️ Duration", f"{summary.get('duration_seconds', 0):.1f}s")
        table.add_row("📄 Pages Crawled", str(summary.get('total_pages_crawled', 0)))
        table.add_row("📋 PDFs Processed", str(summary.get('total_pdfs_processed', 0)))
        table.add_row("💾 Cache Hit Rate", f"{cache_stats.get('hit_rate', 0):.1f}%")
        table.add_row("💾 Total Saved", str(summary.get('total_saved', 0)))
        
        console.print(table)
        
        self.tracker.log_step(SetupStep.DATA_CRAWL, True, "Crawling completed successfully")
        return True
    
    async def _fallback_crawl(self) -> bool:
        """Fallback to basic crawler"""
        console.print("⚠️ Falling back to basic crawler...", style="yellow")
        try:
            from app.crawl.crawler import crawl_pipeline
            
            with Status("🕷️ Running basic crawler...", console=console):
                await crawl_pipeline()
            
            self.tracker.log_step(SetupStep.DATA_CRAWL, True, "Basic crawler completed")
            return True
        except Exception as e:
            self.tracker.log_step(SetupStep.DATA_CRAWL, False, f"Fallback failed: {e}")
            return False

class DataProcessor:
    """Data processing and optimization with Rich display"""
    
    def __init__(self, logger: logging.Logger, tracker: StepTracker):
        self.logger = logger
        self.tracker = tracker
    
    def process_and_optimize_data(self) -> bool:
        """Process and optimize data with cleaning"""
        try:
            data_file = self._find_data_file()
            if not data_file:
                self.tracker.log_step(SetupStep.DATA_PROCESS, False, "No data file found")
                return False
            
            with Status("🧹 Cleaning data...", console=console):
                cleaned_data = self._clean_data(data_file)
            
            with Status("📄 Converting to documents...", console=console):
                documents = self._convert_to_documents(cleaned_data)
            
            if not documents:
                self.tracker.log_step(SetupStep.DATA_PROCESS, False, "No valid documents")
                return False
            
            return self._setup_vector_store(documents)
            
        except Exception as e:
            self.tracker.log_step(SetupStep.DATA_PROCESS, False, str(e))
            return False
    
    def _find_data_file(self) -> Optional[str]:
        """Find available data file"""
        for file_path in ["data/output.json", "data/output.csv"]:
            if os.path.exists(file_path):
                return file_path
        return None
    
    def _clean_data(self, data_file: str) -> List[Dict[str, Any]]:
        """Clean data from file"""
        from app.rag.data_cleaner import DataCleaner
        
        cleaner = DataCleaner()
        
        if hasattr(cleaner, 'clean_data_from_file'):
            return cleaner.clean_data_from_file(data_file)
        elif hasattr(cleaner, 'clean_and_deduplicate_data'):
            import json
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f) if data_file.endswith('.json') else []
            return cleaner.clean_and_deduplicate_data(raw_data)
        else:
            # Use standalone function
            from app.rag.data_cleaner import clean_data_file
            import tempfile
            import json
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                tmp_output = tmp_file.name
            
            clean_data_file(data_file, tmp_output)
            
            with open(tmp_output, 'r', encoding='utf-8') as f:
                cleaned_data = json.load(f)
            
            os.unlink(tmp_output)
            return cleaned_data
    
    def _convert_to_documents(self, cleaned_data: List[Dict[str, Any]]) -> List:
        """Convert cleaned data to Document objects"""
        from langchain.schema import Document
        
        documents = []
        console.print(f"📄 Converting {len(cleaned_data)} items to Document objects...")
        
        for i, item in enumerate(cleaned_data):
            if not isinstance(item, dict):
                continue
            
            content = (item.get('text_content') or item.get('content') or 
                      item.get('text') or item.get('page_content') or '')
            
            if content and len(content.strip()) > 0:
                doc = Document(
                    page_content=content,
                    metadata={
                        'url': item.get('url', ''),
                        'source_type': item.get('source_type', 'unknown'),
                        'title': item.get('title', ''),
                        'content_length': len(content)
                    }
                )
                documents.append(doc)
        
        self.tracker.log_step(
            SetupStep.DATA_PROCESS, 
            True, 
            f"Converted {len(documents)} documents"
        )
        return documents
    
    def _setup_vector_store(self, documents: List) -> bool:
        """Setup vector store with documents"""
        from app.rag.vector_store import VectorStoreManager
        
        try:
            with Status("⚡ Setting up vector store...", console=console):
                vector_manager = VectorStoreManager()
                vector_manager.setup_and_populate_from_documents(documents)
            
            # Create optimized indexes
            try:
                with Status("🔧 Creating performance indexes...", console=console):
                    vector_manager.create_indexes()
                console.print("✅ Performance indexes created", style="green")
            except Exception as index_error:
                console.print(f"⚠️ Index creation failed (non-critical): {index_error}", style="yellow")
            
            return True
            
        except Exception as e:
            self.tracker.log_step(SetupStep.DATA_PROCESS, False, str(e))
            return False

class SystemTester:
    """System testing functionality with Rich output"""
    
    def __init__(self, logger: logging.Logger, tracker: StepTracker):
        self.logger = logger
        self.tracker = tracker
    
    def test_vector_store(self) -> bool:
        """Test vector store functionality"""
        try:
            from app.rag.vector_store import VectorStoreManager
            
            with Status("🧪 Testing vector store...", console=console):
                vector_manager = VectorStoreManager()
                stats = vector_manager.get_vector_store_stats()
            
            if stats.get('document_count', 0) == 0:
                self.tracker.log_step(SetupStep.VECTOR_TEST, False, "No documents in vector store")
                return False
            
            # Test retriever and vector store initialization
            with Status("🔍 Testing retriever...", console=console):
                retriever = vector_manager.get_retriever(
                    search_type="similarity_score_threshold",
                    k=3,
                    score_threshold=0.3
                )
            
            if not retriever:
                self.tracker.log_step(SetupStep.VECTOR_TEST, False, "Failed to initialize retriever")
                return False
            
            with Status("⚡ Testing vector store initialization...", console=console):
                vector_store = vector_manager.initialize_vector_store()
            
            if not vector_store:
                self.tracker.log_step(SetupStep.VECTOR_TEST, False, "Failed to initialize vector store")
                return False
            
            self.tracker.log_step(
                SetupStep.VECTOR_TEST, 
                True, 
                f"Vector store working with {stats['document_count']} documents"
            )
            return True
            
        except Exception as e:
            self.tracker.log_step(SetupStep.VECTOR_TEST, False, str(e))
            return False
    
    def test_system(self) -> bool:
        """Test complete system"""
        try:
            from app.rag.pipeline import create_rag_pipeline
            
            with Status("🔗 Creating RAG pipeline...", console=console):
                pipeline = create_rag_pipeline(enable_cache=True)
            
            # Test connection if available
            if hasattr(pipeline, 'test_connection'):
                with Status("🔌 Testing connection...", console=console):
                    if not pipeline.test_connection():
                        self.tracker.log_step(SetupStep.SYSTEM_TEST, False, "Connection test failed")
                        return False
            
            # Test with sample question
            test_question = "Apa itu Universitas Gunadarma?"
            with Status(f"❓ Testing with question: '{test_question}'...", console=console):
                result = pipeline.ask_question(test_question)
            
            if result and result.get('status') in ['success', 'not_found']:
                self.tracker.log_step(
                    SetupStep.SYSTEM_TEST, 
                    True, 
                    f"Test question processed: {result.get('status')}"
                )
                return True
            else:
                self.tracker.log_step(SetupStep.SYSTEM_TEST, False, "Failed to process test question")
                return False
                
        except Exception as e:
            self.tracker.log_step(SetupStep.SYSTEM_TEST, False, str(e))
            return False

class RAGSystemSetup:
    """Main RAG System Setup orchestrator with Rich interface"""
    
    def __init__(self, config: SetupConfig):
        self.config = config
        self.logger = Logger.setup_logging(config.log_level)
        self.tracker = StepTracker(self.logger)
        self.cache_manager = CacheManager(self.logger)
        self.data_crawler = DataCrawler(self.logger, self.tracker)
        self.data_processor = DataProcessor(self.logger, self.tracker)
        self.system_tester = SystemTester(self.logger, self.tracker)
    
    def check_environment(self) -> bool:
        """Check environment configuration"""
        console.print("🔧 Checking environment configuration...", style="bold blue")
        
        EnvironmentValidator.show_env_status()
        
        is_valid, missing_vars = EnvironmentValidator.validate()
        
        if not is_valid:
            self.tracker.log_step(
                SetupStep.ENVIRONMENT, 
                False, 
                f"Missing variables: {', '.join(missing_vars)}"
            )
            return False
        
        self.tracker.log_step(SetupStep.ENVIRONMENT, True, "All required variables found")
        return True
    
    def setup_database(self) -> bool:
        """Setup database and tables"""
        console.print("🗄️ Setting up database...", style="bold blue")
        
        try:
            from app.rag.db_setup import setup_database
            
            with Status("🗄️ Creating database tables...", console=console):
                setup_database()
            
            self.tracker.log_step(SetupStep.DATABASE, True, "Database tables created")
            return True
        except Exception as e:
            self.tracker.log_step(SetupStep.DATABASE, False, str(e))
            return False
    
    def optimize_vector_store(self) -> bool:
        """Optimize vector store performance"""
        console.print("⚡ Optimizing vector store...", style="bold blue")
        
        try:
            from app.rag.vector_store import VectorStoreManager
            
            vector_manager = VectorStoreManager()
            
            # Cleanup and optimization
            try:
                with Status("🧹 Cleaning up old collections...", console=console):
                    vector_manager.cleanup_old_collections(keep_latest=2)
                
                with Status("🔧 Creating indexes...", console=console):
                    vector_manager.create_indexes()
                
                stats = vector_manager.get_vector_store_stats()
                console.print(f"📊 Vector Store - Collection: {stats['collection_name']}, "
                           f"Documents: {stats['document_count']}", style="green")
                
                self.tracker.log_step(SetupStep.OPTIMIZATION, True, "Performance optimized")
                return True
                
            except Exception as opt_error:
                self.tracker.log_step(SetupStep.OPTIMIZATION, False, str(opt_error))
                return False
                
        except Exception as e:
            self.tracker.log_step(SetupStep.OPTIMIZATION, False, str(e))
            return False
    
    async def run_complete_setup(self) -> bool:
        """Run complete system setup"""
        self._show_welcome_banner()
        self._show_configuration()
        
        try:
            # Step 1: Environment check
            if not self.check_environment():
                return False
            
            # Step 2: Database setup
            if not self.setup_database():
                return False
            
            # Step 3: Cache status
            console.print("📈 Checking cache status...", style="bold blue")
            self.cache_manager.manage_cache("status")
            
            # Step 4: Data crawling
            if not self.config.skip_crawling:
                if not await self.data_crawler.crawl_data(self.config.force_crawl):
                    return False
            else:
                console.print("⏭️ Skipping crawling as requested", style="yellow")
            
            # Step 5: Data processing
            if not self.data_processor.process_and_optimize_data():
                return False
            
            # Step 6: Vector store test
            if not self.system_tester.test_vector_store():
                return False
            
            # Step 7: System test
            if not self.system_tester.test_system():
                return False
            
            # Step 8: Final optimization
            self.optimize_vector_store()
            
            self._show_success_message()
            return True
            
        except KeyboardInterrupt:
            console.print("\n⚠️ Setup interrupted by user", style="yellow")
            return False
        except Exception as e:
            console.print(f"❌ Setup failed: {e}", style="red")
            return False
        finally:
            self.tracker.print_summary()
    
    def _show_welcome_banner(self):
        """Show welcome banner"""
        banner = """
🚀 RAG SYSTEM SETUP
Enhanced with Advanced Caching
        """
        panel = Panel(
            banner.strip(),
            title="Welcome",
            border_style="bold blue",
            padding=(1, 2)
        )
        console.print(panel)
    
    def _show_configuration(self):
        """Show current configuration"""
        config_tree = Tree("⚙️ Configuration")
        config_tree.add(f"Skip crawling: {'✅' if self.config.skip_crawling else '❌'}")
        config_tree.add(f"Force recrawl: {'✅' if self.config.force_crawl else '❌'}")
        config_tree.add("Enhanced caching: ✅")
        config_tree.add(f"Log level: {self.config.log_level}")
        
        console.print(config_tree)
        console.print()
    
    def _show_success_message(self):
        """Show success message with instructions"""
        success_panel = Panel(
            """
🎉 RAG SYSTEM SETUP COMPLETED!

System is ready! You can now:

🚀 [bold cyan]START[/bold cyan] API: [code]uv run uvicorn main:app --reload[/code]
🧪 [bold cyan]TEST[/bold cyan] Run tests: [code]uv run python test/test_system.py[/code]
⚡ [bold cyan]PERF[/bold cyan] Check performance: [code]uv run python test/test_performance.py[/code]
📈 [bold cyan]CACHE[/bold cyan] Cache status: [code]python setup.py cache-status[/code]
🕷️ [bold cyan]CRAWL[/bold cyan] Recrawl data: [code]python setup.py crawl-only[/code]
            """.strip(),
            title="🎯 Success",
            border_style="bold green",
            padding=(1, 2)
        )
        console.print(success_panel)

# Typer CLI Commands

@app.command()
def setup(
    skip_crawling: Annotated[bool, Option("--skip-crawling", help="Skip data crawling, use existing data")] = False,
    force_crawl: Annotated[bool, Option("--force-crawl", help="Force recrawl even if data exists")] = False,
    log_level: Annotated[str, Option("--log-level", help="Set logging level")] = "INFO"
):
    """🚀 Run complete RAG system setup"""
    config = SetupConfig(
        skip_crawling=skip_crawling,
        force_crawl=force_crawl,
        log_level=log_level
    )
    
    setup_instance = RAGSystemSetup(config)
    success = asyncio.run(setup_instance.run_complete_setup())
    
    if not success:
        raise typer.Exit(1)

@app.command("crawl-only")
def crawl_only(
    force_crawl: Annotated[bool, Option("--force", help="Force recrawl even if data exists")] = False,
    log_level: Annotated[str, Option("--log-level", help="Set logging level")] = "INFO"
):
    """🕷️  Run crawling only, skip other steps"""
    config = SetupConfig(crawl_only=True, force_crawl=force_crawl, log_level=log_level)
    setup_instance = RAGSystemSetup(config)
    
    async def run_crawl():
        return await setup_instance.data_crawler.crawl_data(force_crawl)
    
    success = asyncio.run(run_crawl())
    
    if not success:
        raise typer.Exit(1)

@app.command("cache-status")
def cache_status():
    """📈 Show cache statistics"""
    logger = Logger.setup_logging("INFO")
    cache_manager = CacheManager(logger)
    cache_manager.manage_cache("status")

@app.command("cache-cleanup")
def cache_cleanup():
    """🧹 Clean up expired cache entries"""
    logger = Logger.setup_logging("INFO")
    cache_manager = CacheManager(logger)
    
    if not cache_manager.manage_cache("cleanup"):
        raise typer.Exit(1)

@app.command("cache-clear")
def cache_clear():
    """🗑️  Clear all cache data"""
    logger = Logger.setup_logging("INFO")
    cache_manager = CacheManager(logger)
    
    if not cache_manager.manage_cache("clear"):
        raise typer.Exit(1)

@app.command("optimize-only")
def optimize_only(
    log_level: Annotated[str, Option("--log-level", help="Set logging level")] = "INFO"
):
    """⚡ Optimize vector store indexes only"""
    config = SetupConfig(optimize_only=True, log_level=log_level)
    setup_instance = RAGSystemSetup(config)
    
    if not setup_instance.optimize_vector_store():
        raise typer.Exit(1)

@app.command("env-check")
def env_check():
    """🔧 Check environment variables"""
    EnvironmentValidator.show_env_status()
    
    is_valid, missing_vars = EnvironmentValidator.validate()
    
    if not is_valid:
        console.print(f"\n❌ Missing variables: {', '.join(missing_vars)}", style="red")
        raise typer.Exit(1)
    else:
        console.print("\n✅ All environment variables are set!", style="green")

if __name__ == "__main__":
    app()