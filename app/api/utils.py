"""
Utility functions untuk API
"""
import time
import hashlib
from typing import Any, Dict, List, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def generate_cache_key(question: str, metadata_filter: Optional[Dict[str, Any]] = None) -> str:
    """Generate cache key dari question dan metadata filter"""
    cache_data = {
        "question": question.strip().lower(),
        "metadata_filter": metadata_filter or {}
    }
    
    # Convert to string and hash
    cache_string = str(sorted(cache_data.items()))
    return hashlib.md5(cache_string.encode()).hexdigest()


def format_response_time(start_time: float) -> float:
    """Format response time dengan precision 3 decimal places"""
    return round(time.time() - start_time, 3)


def sanitize_question(question: str) -> str:
    """Sanitize question input"""
    if not question:
        return ""
    
    # Remove extra whitespace
    question = question.strip()
    
    # Remove multiple spaces
    question = ' '.join(question.split())
    
    # Basic sanitization (remove potentially harmful characters)
    # In production, you might want more sophisticated sanitization
    dangerous_chars = ['<', '>', '"', "'", '&', '\n', '\r', '\t']
    for char in dangerous_chars:
        question = question.replace(char, ' ')
    
    return question


def measure_execution_time(func):
    """Decorator untuk mengukur execution time"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = format_response_time(start_time)
            logger.info(f"{func.__name__} executed in {execution_time}s")
            return result
        except Exception as e:
            execution_time = format_response_time(start_time)
            logger.error(f"{func.__name__} failed in {execution_time}s: {str(e)}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = format_response_time(start_time)
            logger.info(f"{func.__name__} executed in {execution_time}s")
            return result
        except Exception as e:
            execution_time = format_response_time(start_time)
            logger.error(f"{func.__name__} failed in {execution_time}s: {str(e)}")
            raise
    
    return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper


def validate_url_list(urls: List[str]) -> List[str]:
    """Validate dan sanitize list of URLs"""
    validated_urls = []
    for url in urls:
        if isinstance(url, str) and url.strip():
            # Basic URL validation
            if url.startswith(('http://', 'https://')):
                validated_urls.append(url.strip())
    return validated_urls


def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def extract_domain_from_url(url: str) -> str:
    """Extract domain dari URL"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return url


def format_source_urls(source_urls: List[str]) -> List[str]:
    """Format dan validate source URLs"""
    if not source_urls:
        return []
    
    formatted_urls = []
    for url in source_urls:
        if isinstance(url, str) and url.strip():
            # Ensure URL is properly formatted
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            formatted_urls.append(url)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in formatted_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls


def create_error_response(message: str, error_code: str = "INTERNAL_ERROR") -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "error": True,
        "error_code": error_code,
        "message": message,
        "timestamp": time.time()
    }


def log_performance_metrics(
    operation: str,
    execution_time: float,
    additional_metrics: Optional[Dict[str, Any]] = None
):
    """Log performance metrics"""
    metrics = {
        "operation": operation,
        "execution_time": execution_time,
        "timestamp": time.time()
    }
    
    if additional_metrics:
        metrics.update(additional_metrics)
    
    logger.info(f"Performance metrics: {metrics}")