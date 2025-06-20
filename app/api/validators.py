"""
Custom validators untuk request validation
"""
from typing import List
from pydantic import validator

from app.api.config import settings


class QuestionValidator:
    """Validator untuk question input"""
    
    @staticmethod
    def validate_question_not_empty(question: str) -> str:
        """Validate that question is not empty"""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        return question.strip()
    
    @staticmethod
    def validate_question_length(question: str, max_length: int = 1000) -> str:
        """Validate question length"""
        if len(question) > max_length:
            raise ValueError(f"Question too long. Maximum {max_length} characters allowed")
        return question


class BatchValidator:
    """Validator untuk batch requests"""
    
    @staticmethod
    def validate_batch_size(questions: List[str]) -> List[str]:
        """Validate batch size"""
        if not questions:
            raise ValueError("Questions list cannot be empty")
        
        if len(questions) > settings.MAX_BATCH_SIZE:
            raise ValueError(f"Maximum {settings.MAX_BATCH_SIZE} questions per batch")
        
        return questions
    
    @staticmethod
    def validate_questions_not_empty(questions: List[str]) -> List[str]:
        """Validate that all questions in batch are not empty"""
        validated_questions = []
        for i, question in enumerate(questions):
            if not question or not question.strip():
                raise ValueError(f"Question at index {i} cannot be empty")
            validated_questions.append(question.strip())
        return validated_questions


class MetadataValidator:
    """Validator untuk metadata filters"""
    
    @staticmethod
    def validate_metadata_filter(metadata_filter: dict) -> dict:
        """Validate metadata filter structure"""
        if not isinstance(metadata_filter, dict):
            raise ValueError("Metadata filter must be a dictionary")
        
        # Add specific validation rules based on your metadata structure
        allowed_keys = ["source", "category", "type", "date_range"]
        
        for key in metadata_filter.keys():
            if key not in allowed_keys:
                raise ValueError(f"Invalid metadata key: {key}. Allowed keys: {allowed_keys}")
        
        return metadata_filter