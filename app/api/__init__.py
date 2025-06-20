"""
API module untuk RAG chatbot system

Module ini berisi semua komponen API termasuk FastAPI app factory,
models, routes, middleware, dan utilities.
"""

from .app import create_app
from .config import settings
from .models import (
    QuestionRequest,
    QuestionResponse,
    BatchQuestionRequest,
    BatchQuestionResponse,
    HealthResponse,
    StatsResponse,
    CacheResponse,
    ExampleQuestionsResponse
)
from .exceptions import (
    RAGPipelineNotInitialized,
    ApplicationStartupIncomplete,
    EmptyQuestionError,
    EmptyQuestionListError,
    BatchSizeLimitError,
    CacheNotAvailableError,
    InternalServerError,
    HealthCheckFailedError
)
from .dependencies import AppState

__all__ = [
    # App factory
    'create_app',
    
    # Configuration
    'settings',
    
    # Models
    'QuestionRequest',
    'QuestionResponse',
    'BatchQuestionRequest',
    'BatchQuestionResponse',
    'HealthResponse',
    'StatsResponse',
    'CacheResponse',
    'ExampleQuestionsResponse',
      # Exceptions
    'RAGPipelineNotInitialized',
    'ApplicationStartupIncomplete',
    'EmptyQuestionError',
    'EmptyQuestionListError',
    'BatchSizeLimitError',
    'CacheNotAvailableError',
    'InternalServerError',
    'HealthCheckFailedError',
    
    # Dependencies
    'AppState'
]
