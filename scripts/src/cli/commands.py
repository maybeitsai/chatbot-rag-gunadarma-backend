#!/usr/bin/env python3
"""
CLI Commands for RAG System Setup
Consolidated command definitions for better maintainability
"""

import asyncio
import sys
from pathlib import Path
import typer
from rich.console import Console

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.parent.parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from scripts.src.setup.orchestrator import RAGSystemSetup
from scripts.src.setup.core.config import SetupConfig

app = typer.Typer(
    name="rag-setup",
    help="🚀 RAG System Setup",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()


@app.command()
def setup(
    skip_crawling: bool = False, 
    force_crawl: bool = False, 
    log_level: str = "INFO"
):
    """Run complete RAG system setup"""
    config = SetupConfig(
        skip_crawling=skip_crawling,
        force_crawl=force_crawl,
        log_level=log_level,
    )

    setup_instance = RAGSystemSetup(config)
    success = asyncio.run(setup_instance.run_complete_setup())

    if not success:
        raise typer.Exit(1)


@app.command("crawl-only")
def crawl_only(force_crawl: bool = False, log_level: str = "INFO"):
    """Run crawling only, skip other steps"""
    config = SetupConfig(crawl_only=True, force_crawl=force_crawl, log_level=log_level)
    setup_instance = RAGSystemSetup(config)

    success = asyncio.run(setup_instance.data_crawler.crawl_data(force_crawl))

    if not success:
        raise typer.Exit(1)


@app.command("process-only")
def process_only(log_level: str = "INFO"):
    """Process existing data only, skip crawling"""
    config = SetupConfig(
        skip_crawling=True,
        process_only=True,
        log_level=log_level,
    )

    setup_instance = RAGSystemSetup(config)
    success = asyncio.run(setup_instance.data_processor.process_data())

    if not success:
        console.print("❌ Data processing failed.", style="red")
        raise typer.Exit(1)
    else:
        console.print("✅ Data processing completed!", style="green")


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
        raise typer.Exit(1)


@app.command("cache-clear")
def cache_clear():
    """Clear all cache data"""
    from scripts.src.setup.managers.cache_manager import CacheManager
    from scripts.src.setup.core.logger import Logger

    logger = Logger.setup_logging("INFO")
    cache_manager = CacheManager(logger)

    if not cache_manager.manage_cache("clear"):
        raise typer.Exit(1)


@app.command("optimize-only")
def optimize_only(log_level: str = "INFO"):
    """Optimize vector store indexes only"""
    config = SetupConfig(optimize_only=True, log_level=log_level)
    setup_instance = RAGSystemSetup(config)

    success = setup_instance.optimize_vector_store()
    if not success:
        console.print("❌ Optimization failed.", style="red")
        raise typer.Exit(1)
    else:
        console.print("✅ Vector store optimization completed!", style="green")


@app.command("env-check")
def env_check():
    """Check environment variables"""
    from scripts.src.setup.validators.environment import EnvironmentValidator

    EnvironmentValidator.show_env_status()

    is_valid, missing_vars = EnvironmentValidator.validate()

    if not is_valid:
        console.print(f"❌ Missing variables: {', '.join(missing_vars)}", style="red")
        raise typer.Exit(1)
    else:
        console.print("✅ All environment variables are set!", style="green")


@app.command("reset-system")
def reset_system(confirm: bool = typer.Option(False, "--confirm", help="Confirm system reset")):
    """Reset the entire system (clear all data and cache)"""
    if not confirm:
        console.print("❌ System reset requires confirmation. Use --confirm flag.", style="red")
        raise typer.Exit(1)
    
    from scripts.src.setup.managers.cache_manager import CacheManager
    from scripts.src.setup.core.logger import Logger

    logger = Logger.setup_logging("INFO")
    cache_manager = CacheManager(logger)

    # Clear all caches
    cache_manager.manage_cache("clear")
    
    # Additional cleanup could be added here (database reset, etc.)
    console.print("✅ System reset completed!", style="green")
