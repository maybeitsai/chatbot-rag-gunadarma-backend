#!/usr/bin/env python3
"""
CLI Application for RAG System Setup
"""

import sys
from pathlib import Path
from typer import Typer

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.parent.parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from scripts.src.setup.orchestrator import RAGSystemSetup
from scripts.src.setup.core.config import SetupConfig

app = Typer(
    name="rag-setup",
    help="🚀 RAG System Setup - Enhanced with Advanced Caching",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def setup(
    skip_crawling: bool = False, force_crawl: bool = False, log_level: str = "INFO"
):
    """Run complete RAG system setup"""
    import asyncio

    config = SetupConfig(
        skip_crawling=skip_crawling, force_crawl=force_crawl, log_level=log_level
    )

    setup_instance = RAGSystemSetup(config)
    success = asyncio.run(setup_instance.run_complete_setup())

    if not success:
        raise SystemExit(1)


@app.command("crawl-only")
def crawl_only(force_crawl: bool = False, log_level: str = "INFO"):
    """Run crawling only, skip other steps"""
    import asyncio

    config = SetupConfig(crawl_only=True, force_crawl=force_crawl, log_level=log_level)
    setup_instance = RAGSystemSetup(config)

    success = asyncio.run(setup_instance.data_crawler.crawl_data(force_crawl))

    if not success:
        raise SystemExit(1)


@app.command("cache-status")
def cache_status():
    """Show cache statistics"""
    from scripts.src.setup.managers.cache_manager import CacheManager
    from scripts.src.setup.core.logger import Logger

    logger = Logger.setup_logging("INFO")
    cache_manager = CacheManager(logger)
    cache_manager.manage_cache("status")


@app.command("cache-cleanup")
def cache_cleanup():
    """Clean up expired cache entries"""
    from scripts.src.setup.managers.cache_manager import CacheManager
    from scripts.src.setup.core.logger import Logger

    logger = Logger.setup_logging("INFO")
    cache_manager = CacheManager(logger)

    if not cache_manager.manage_cache("cleanup"):
        raise SystemExit(1)


@app.command("cache-clear")
def cache_clear():
    """Clear all cache data"""
    from scripts.src.setup.managers.cache_manager import CacheManager
    from scripts.src.setup.core.logger import Logger

    logger = Logger.setup_logging("INFO")
    cache_manager = CacheManager(logger)

    if not cache_manager.manage_cache("clear"):
        raise SystemExit(1)


@app.command("optimize-only")
def optimize_only(log_level: str = "INFO"):
    """Optimize vector store indexes only"""
    config = SetupConfig(optimize_only=True, log_level=log_level)
    setup_instance = RAGSystemSetup(config)

    if not setup_instance.optimize_vector_store():
        raise SystemExit(1)


@app.command("env-check")
def env_check():
    """Check environment variables"""
    from scripts.src.setup.validators.environment import EnvironmentValidator

    EnvironmentValidator.show_env_status()

    is_valid, missing_vars = EnvironmentValidator.validate()

    if not is_valid:
        print(f"Missing variables: {', '.join(missing_vars)}")
        raise SystemExit(1)
    else:
        print("All environment variables are set!")
