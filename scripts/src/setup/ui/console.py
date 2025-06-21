from rich.console import Console
from rich.prompt import Prompt

class ConsoleManager:
    def __init__(self):
        self.console = Console()

    def print_message(self, message: str, style: str = "white"):
        self.console.print(message, style=style)

    def prompt_confirmation(self, question: str, default: bool = True) -> bool:
        return Prompt.ask(f"{question} [y/n]", default="y" if default else "n").lower() == "y"

    def prompt_input(self, question: str, default: str = "") -> str:
        return Prompt.ask(question, default=default)

    def print_success(self, message: str):
        self.print_message(f"[green]✔️ {message}[/green]")

    def print_error(self, message: str):
        self.print_message(f"[red]✖️ {message}[/red]")

    def print_warning(self, message: str):
        self.print_message(f"[yellow]⚠️ {message}[/yellow]")