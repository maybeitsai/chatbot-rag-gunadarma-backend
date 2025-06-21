from pathlib import Path
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table


class CacheManager:
    """Manage cache operations during the setup process."""

    def __init__(self, logger):
        self.logger = logger
        self.console = Console()

    def manage_cache(self, action: str) -> bool:
        """Manage cache operations based on the action specified."""
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
            self.console.print("⚠️ Enhanced cache manager not available", style="yellow")
            return False
        except Exception as e:
            self.console.print(f"❌ Cache management failed: {e}", style="red")
            return False

    def _show_cache_status(self, cache_manager) -> bool:
        """Show cache status with a table."""
        stats = cache_manager.get_cache_statistics()

        table = Table(
            title="📈 Cache Statistics", show_header=True, header_style="bold green"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Total Requests", str(stats["total_requests"]))
        table.add_row("Hit Rate", f"{stats['hit_rate']:.1f}%")
        table.add_row("Total Entries", str(sum(stats["cache_sizes"].values())))

        self.console.print(table)

        if stats["cache_sizes"]:
            detail_table = Table(
                title="📋 Cache Details", show_header=True, header_style="bold blue"
            )
            detail_table.add_column("Cache Type", style="cyan")
            detail_table.add_column("Entries", justify="right", style="green")

            for cache_type, size in stats["cache_sizes"].items():
                if size > 0:
                    detail_table.add_row(
                        cache_type.replace("_", " ").title(), str(size)
                    )

            self.console.print(detail_table)

        return True

    def _cleanup_cache(self, cache_manager) -> bool:
        """Cleanup expired cache entries."""
        with self.console.status("🧹 Cleaning up expired cache entries..."):
            cache_manager.cleanup_expired_cache()

        self.console.print("✅ Expired cache entries removed", style="green")
        return True

    def _clear_cache(self, cache_manager) -> bool:
        """Clear all cache with confirmation."""
        if not Confirm.ask("🗑️ Are you sure you want to clear ALL cache data?"):
            self.console.print("ℹ️ Cache clear cancelled", style="yellow")
            return False

        with self.console.status("🗑️ Clearing all cache data..."):
            for cache_attr in [
                "url_cache",
                "content_cache",
                "response_cache",
                "similarity_cache",
                "robots_cache",
            ]:
                if hasattr(cache_manager, cache_attr):
                    getattr(cache_manager, cache_attr).clear()

            cache_dir = Path("cache")
            if cache_dir.exists():
                cache_files = [
                    "url_cache.json",
                    "response_cache.json",
                    "similarity_cache.json",
                    "robots_cache.json",
                ]
                for cache_file in cache_files:
                    cache_path = cache_dir / cache_file
                    cache_path.unlink(missing_ok=True)

        self.console.print("✅ All cache data cleared", style="green")
        return True
