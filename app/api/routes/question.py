"""
Routes untuk question handling
"""
from fastapi import APIRouter, Depends

from app.api.models import (
    QuestionRequest, 
    QuestionResponse, 
    BatchQuestionRequest, 
    BatchQuestionResponse
)
from app.api.dependencies import get_rag_pipeline
from app.api.services import RAGService
from app.api.exceptions import EmptyQuestionError, EmptyQuestionListError, BatchSizeLimitError

router = APIRouter(prefix="/api/v1/ask", tags=["questions"])


@router.post("", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    rag_pipeline=Depends(get_rag_pipeline)
):
    """Enhanced endpoint untuk mengajukan pertanyaan dengan dukungan async, caching, dan hybrid search"""
    if not request.question.strip():
        raise EmptyQuestionError()
    
    return await RAGService.process_question(
        rag_pipeline=rag_pipeline,
        question=request.question,
        metadata_filter=request.metadata_filter,
        use_cache=request.use_cache,
        use_hybrid=request.use_hybrid
    )


@router.post("/api/v1/batch", response_model=BatchQuestionResponse)
async def ask_questions_batch(
    request: BatchQuestionRequest,
    rag_pipeline=Depends(get_rag_pipeline)
):
    """Endpoint untuk mengajukan multiple pertanyaan secara batch"""
    if not request.questions:
        raise EmptyQuestionListError()
    
    if len(request.questions) > 10:  # Limit batch size
        raise BatchSizeLimitError(max_size=10)    
    results, processing_time = await RAGService.process_batch_questions(
        rag_pipeline=rag_pipeline,
        questions=request.questions,
        use_cache=request.use_cache,
        use_hybrid=request.use_hybrid
    )
    
    return BatchQuestionResponse(
        results=results,
        total_questions=len(request.questions),
        processing_time=processing_time
    )