import logging
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console


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
                RichHandler(console=Console(), rich_tracebacks=True),
                logging.FileHandler(log_dir / "setup.log", encoding="utf-8"),
            ],
        )
        return logging.getLogger(__name__)
