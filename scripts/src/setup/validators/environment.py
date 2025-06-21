import os
from typing import List, Tuple

class EnvironmentValidator:
    REQUIRED_ENV_VARS = [
        "GOOGLE_API_KEY",
        "NEON_CONNECTION_STRING", 
        "LLM_MODEL",
        "EMBEDDING_MODEL"
    ]
    
    @classmethod
    def validate(cls) -> Tuple[bool, List[str]]:
        missing_vars = [var for var in cls.REQUIRED_ENV_VARS if not os.getenv(var)]
        return len(missing_vars) == 0, missing_vars
    
    @classmethod
    def show_env_status(cls):
        from rich.table import Table
        from rich.console import Console
        
        console = Console()
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