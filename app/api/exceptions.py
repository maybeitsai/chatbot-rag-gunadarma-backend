"""
Custom exceptions untuk API
"""
from fastapi import HTTPException


class RAGPipelineNotInitialized(HTTPException):
    def __init__(self):
        super().__init__(status_code=503, detail="RAG Pipeline not initialized")


class ApplicationStartupIncomplete(HTTPException):
    def __init__(self):
        super().__init__(status_code=503, detail="Application is still starting up")


class EmptyQuestionError(HTTPException):
    def __init__(self):
        super().__init__(status_code=400, detail="Question cannot be empty")


class EmptyQuestionListError(HTTPException):
    def __init__(self):
        super().__init__(status_code=400, detail="Questions list cannot be empty")


class BatchSizeLimitError(HTTPException):
    def __init__(self, max_size: int = 10):
        super().__init__(
            status_code=400, 
            detail=f"Maximum {max_size} questions per batch"
        )


class CacheNotAvailableError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=404, 
            detail="Cache not available in current pipeline"
        )


class InternalServerError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=f"Internal server error: {detail}")


class HealthCheckFailedError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=503, detail=f"Health check failed: {detail}")