from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


class Display:
    @staticmethod
    def show_welcome_banner():
        banner = """
🚀 RAG SYSTEM SETUP
Enhanced with Advanced Caching
        """
        panel = Panel(
            banner.strip(), title="Welcome", border_style="bold blue", padding=(1, 2)
        )
        console.print(panel)

    @staticmethod
    def show_configuration(config):
        config_tree = Table(title="⚙️ Configuration", show_header=False)
        config_tree.add_column("Setting", style="cyan")
        config_tree.add_column("Value", style="green")

        config_tree.add_row("Skip crawling", "✅" if config.skip_crawling else "❌")
        config_tree.add_row("Force recrawl", "✅" if config.force_crawl else "❌")
        config_tree.add_row("Enhanced caching", "✅")
        config_tree.add_row("Log level", config.log_level)

        console.print(config_tree)
        console.print()

    @staticmethod
    def show_success_message():
        success_panel = Panel(
            """
🎉 RAG SYSTEM SETUP COMPLETED!

System is ready! You can now:

🚀 [bold cyan]START[/bold cyan] API: [code]uvicorn main:app --reload[/code]
🧪 [bold cyan]TEST[/bold cyan] Run tests: [code]python tests/test_system.py[/code]
⚡ [bold cyan]PERF[/bold cyan] Check performance: [code]python tests/test_performance.py[/code]
📈 [bold cyan]CACHE[/bold cyan] Cache status: [code]python scripts/run.py cache-status[/code]
🕷️  [bold cyan]CRAWL[/bold cyan] Recrawl data: [code]python scripts/run.py crawl-only[/code]
            """.strip(),
            title="🎯 Success",
            border_style="bold green",
            padding=(1, 2),
        )
        console.print(success_panel)

    @staticmethod
    def check_cache_status(cache_manager):
        """Display cache status"""
        console.print("📈 Checking cache status...")
        cache_manager.manage_cache("status")

    @staticmethod
    def skip_crawling():
        """Show skip crawling message"""
        console.print("⏭️ Skipping crawling (using existing data)", style="yellow")

    @staticmethod
    def check_environment_status():
        """Show environment check header"""
        console.print("🔧 Checking environment variables...")


# Backward compatibility - keep the original functions
def show_welcome_banner():
    Display.show_welcome_banner()


def show_configuration(config):
    Display.show_configuration(config)


def show_success_message():
    Display.show_success_message()
