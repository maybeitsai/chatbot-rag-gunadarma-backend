from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from rag.pipeline import RAGPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Gunadarma RAG API",
    description="Retrieval-Augmented Generation API untuk informasi Universitas Gunadarma",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    source_urls: List[str]
    status: str
    source_count: Optional[int] = 0

# Initialize RAG pipeline
try:
    rag_pipeline = RAGPipeline()
    print("RAG Pipeline initialized successfully!")
except Exception as e:
    print(f"Failed to initialize RAG Pipeline: {e}")
    rag_pipeline = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Gunadarma RAG API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    # Test connection
    is_healthy = rag_pipeline.test_connection()
    
    if not is_healthy:
        raise HTTPException(status_code=503, detail="RAG Pipeline connection failed")
    
    return {
        "status": "healthy",
        "rag_pipeline": "connected",
        "database": "connected"
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Main endpoint untuk mengajukan pertanyaan"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Process question through RAG pipeline
        result = rag_pipeline.ask_question(request.question)
        
        return QuestionResponse(
            answer=result["answer"],
            source_urls=result["source_urls"],
            status=result["status"],
            source_count=result.get("source_count", 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        # Get vector store statistics
        vector_store = rag_pipeline.vector_store
        
        return {
            "embedding_model": os.getenv("EMBEDDING_MODEL"),
            "llm_model": os.getenv("LLM_MODEL"),
            "chunk_size": int(os.getenv("CHUNK_SIZE", 500)),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 50)),
            "collection_name": rag_pipeline.vector_store_manager.collection_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Example questions endpoint
@app.get("/examples")
async def get_example_questions():
    """Get example questions that can be asked"""
    return {
        "example_questions": [
            "Apa itu Universitas Gunadarma?",
            "Fakultas apa saja yang ada di Universitas Gunadarma?",
            "Bagaimana cara mendaftar di Universitas Gunadarma?",
            "Apa saja program studi yang tersedia?",
            "Dimana lokasi kampus Universitas Gunadarma?",
            "Bagaimana sistem pembelajaran di Universitas Gunadarma?",
            "Apa saja fasilitas yang tersedia di kampus?",
            "Bagaimana cara menghubungi BAAK Universitas Gunadarma?"
        ]
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )