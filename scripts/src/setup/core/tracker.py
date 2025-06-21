import logging
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from typing import List, Union
import time
from scripts.src.setup.core.enums import SetupStep


class StepTracker:
    """Track setup steps and results with Rich progress"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = time.time()
        self.completed_steps: List[str] = []
        self.errors: List[str] = []
        self.current_progress = None
        self.console = Console()

    def log_step(
        self, step: Union[SetupStep, str], success: bool = True, details: str = ""
    ):
        """Log setup step completion with Rich formatting"""
        if success:
            status_icon = "✅"
            status_color = "green"
        else:
            status_icon = "❌"
            status_color = "red"

        # Handle both SetupStep enum and string
        step_name = step.value if isinstance(step, SetupStep) else step
        step_text = Text(f"{status_icon} {step_name}", style=status_color)
        self.console.print(step_text)

        if details:
            detail_text = Text(f"   {details}", style="dim")
            self.console.print(detail_text)

        if success:
            self.completed_steps.append(step_name)
        else:
            self.errors.append(f"{step_name}: {details}")

    def get_duration(self) -> float:
        """Get total duration"""
        return time.time() - self.start_time

    def print_summary(self):
        """Print setup summary with Rich table"""
        duration = self.get_duration()

        table = Table(
            title="📊 Setup Summary", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        table.add_row("⏱️ Duration", f"{duration:.1f} seconds")
        table.add_row("✅ Steps Completed", str(len(self.completed_steps)))
        table.add_row("❌ Errors", str(len(self.errors)))

        self.console.print()
        self.console.print(table)

        if self.errors:
            error_panel = Panel(
                "\n".join([f"• {error}" for error in self.errors]),
                title="❌ Errors Encountered",
                border_style="red",
            )
            self.console.print(error_panel)
