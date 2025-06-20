"""
Entry point untuk Gunadarma RAG API
"""
import uvicorn

from app.api.app import create_app
from app.api.config import settings

# Create FastAPI app instance
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )