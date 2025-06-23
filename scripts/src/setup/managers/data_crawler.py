from pathlib import Path
import os
import json
from typing import List, Dict, Any
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.status import Status
from app.crawl.config import CrawlConfig
from app.crawl.crawler import WebCrawler
from rich.prompt import Confirm
from rich.panel import Panel
from rich.table import Table
import logging

from scripts.src.setup.core.tracker import StepTracker, SetupStep


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
                f"Found existing data: {', '.join(existing_files)}",
            )

            if not self._should_recrawl():
                return True

        return await self._perform_crawl()

    def _should_recrawl(self) -> bool:
        """Ask user if they want to recrawl with Rich prompt"""
        return Confirm.ask(
            "\n🔄 Existing data found. Recrawl with enhanced caching?", default=False
        )

    async def _perform_crawl(self) -> bool:
        """Perform the actual crawling with Rich progress"""
        Console().print("🕷️ Starting enhanced crawling...", style="bold blue")

        try:
            target_urls = [
                "https://baak.gunadarma.ac.id/",
                "https://www.gunadarma.ac.id/",
            ]
            config = self._get_crawl_config()
            self._show_crawl_config(config, target_urls)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=Console(),
            ) as progress:
                task = progress.add_task("🕷️ Crawling websites...", total=100)

                crawler = WebCrawler(target_urls, config)
                report = await crawler.crawl(incremental=True)

                progress.update(task, completed=100)

            return self._process_crawl_report(report)

        except ImportError:
            return await self._fallback_crawl()
        except Exception as e:
            Console().print(f"❌ Enhanced crawling failed: {e}", style="red")
            return await self._fallback_crawl()

    def _get_crawl_config(self):
        """Get optimized crawl configuration"""
        return CrawlConfig(
            max_depth=3,
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
            timeout=60,
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
            config_text, title="🚀 Enhanced Crawling Configuration", border_style="blue"
        )
        Console().print(panel)

    def _process_crawl_report(self, report: Dict[str, Any]) -> bool:
        """Process crawling report with Rich table"""
        if report.get("status") != "success":
            error_msg = report.get("error", "Unknown crawling error")
            self.tracker.log_step(SetupStep.DATA_CRAWL, False, error_msg)
            return False

        summary = report["crawl_summary"]
        cache_stats = report.get("cache_statistics", {})

        # Create results table
        table = Table(
            title="📈 Crawling Results", show_header=True, header_style="bold green"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("⏱️ Duration", f"{summary.get('duration_seconds', 0):.1f}s")
        table.add_row("📄 Pages Crawled", str(summary.get("total_pages_crawled", 0)))
        table.add_row("📋 PDFs Processed", str(summary.get("total_pdfs_processed", 0)))
        table.add_row("💾 Cache Hit Rate", f"{cache_stats.get('hit_rate', 0):.1f}%")
        table.add_row("💾 Total Saved", str(summary.get("total_saved", 0)))

        Console().print(table)

        self.tracker.log_step(
            SetupStep.DATA_CRAWL, True, "Crawling completed successfully"
        )
        return True

    async def _fallback_crawl(self) -> bool:
        """Fallback to basic crawler"""
        Console().print("⚠️ Falling back to basic crawler...", style="yellow")
        try:
            from app.crawl.crawler import crawl_pipeline

            with Status("🕷️ Running basic crawler...", console=Console()):
                await crawl_pipeline()

            self.tracker.log_step(SetupStep.DATA_CRAWL, True, "Basic crawler completed")
            return True
        except Exception as e:
            self.tracker.log_step(SetupStep.DATA_CRAWL, False, f"Fallback failed: {e}")
            return False
