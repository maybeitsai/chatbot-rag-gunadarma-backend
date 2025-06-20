"""
Pydantic models untuk request dan response API
"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class QuestionRequest(BaseModel):
    question: str
    use_cache: Optional[bool] = True
    metadata_filter: Optional[Dict[str, Any]] = None


class QuestionResponse(BaseModel):
    answer: str
    source_urls: List[str]
    status: str
    source_count: Optional[int] = 0
    response_time: Optional[float] = None
    cached: Optional[bool] = False
    cache_type: Optional[str] = None


class BatchQuestionRequest(BaseModel):
    questions: List[str]
    use_cache: Optional[bool] = True


class BatchQuestionResponse(BaseModel):
    results: List[QuestionResponse]
    total_questions: int
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    rag_pipeline: str
    database: str
    cache_enabled: bool
    optimized: bool
    stats: Optional[Dict[str, Any]] = None


class StatsResponse(BaseModel):
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


class CacheResponse(BaseModel):
    message: str


class ExampleQuestionsResponse(BaseModel):
    example_questions: List[str]