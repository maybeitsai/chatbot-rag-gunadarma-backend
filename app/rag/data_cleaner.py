"""
Data cleaning and preprocessing utilities
Handles HTML cleaning, URL validation, and content deduplication
"""

import re
import html
import hashlib
from typing import List, Dict, Set, Optional
from urllib.parse import unquote, quote
from difflib import SequenceMatcher
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataCleaner:
    """Enhanced data cleaning utilities with optimized performance"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.seen_content_hashes: Set[str] = set()
        self.url_content_map: Dict[str, str] = {}
        self.content_fingerprints: Dict[str, List[str]] = defaultdict(list)
        
        # Compiled regex patterns for better performance
        self.html_tag_pattern = re.compile(r'<[^>]+>', re.IGNORECASE)
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\w\s\-.,!?()[\]{}";:\'/\\@#$%^&*+=<>~`|]')
        self.url_spaces_pattern = re.compile(r'\s+')
        self.sentence_pattern = re.compile(r'[.!?]+\s*')
        
    def clean_html_content(self, text: str) -> str:
        """
        Remove HTML tags and clean up text content with optimized performance
        
        Args:
            text: Raw HTML text content
            
        Returns:
            Cleaned text without HTML tags and normalized whitespace
        """
        if not text:
            return ""
            
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = self.html_tag_pattern.sub(' ', text)
        
        # Normalize whitespace in one pass
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def validate_and_fix_url(self, url: str) -> str:
        """
        Validate and fix URL formatting with optimized performance
        
        Args:
            url: URL to validate and fix
            
        Returns:
            Properly formatted URL with encoded spaces
        """
        if not url:
            return ""
            
        # Remove extra whitespace and handle spaces in one operation
        url = url.strip().replace(' ', '%20')
        
        return url
    
    def calculate_content_hash(self, content: str) -> str:
        """
        Calculate hash for content deduplication with faster normalization
        
        Args:
            content: Text content to hash
            
        Returns:
            SHA256 hash of normalized content
        """
        # Faster normalization using compiled regex
        normalized = self.whitespace_pattern.sub(' ', content.lower().strip())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def create_content_fingerprint(self, content: str) -> str:
        """
        Create a faster fingerprint for content similarity detection
        Uses first and last sentences plus content length
        
        Args:
            content: Text content to fingerprint
            
        Returns:
            Content fingerprint for quick similarity comparison
        """
        sentences = self.sentence_pattern.split(content.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return f"len:{len(content)}"
        
        # Create fingerprint from first sentence, last sentence, and length
        first_sentence = sentences[0][:100] if sentences else ""
        last_sentence = sentences[-1][:100] if len(sentences) > 1 else ""
        
        fingerprint = f"{first_sentence}|{last_sentence}|len:{len(content)}"
        return hashlib.md5(fingerprint.encode('utf-8')).hexdigest()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts with early exit optimization
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity ratio between 0 and 1
        """
        # Quick length-based filtering
        len1, len2 = len(text1), len(text2)
        if abs(len1 - len2) / max(len1, len2, 1) > 0.3:
            return 0.0
        
        # Use only first 500 characters for similarity comparison to speed up
        sample1 = text1[:500].lower()
        sample2 = text2[:500].lower()
        
        return SequenceMatcher(None, sample1, sample2).ratio()
    
    def is_duplicate_content(self, content: str, url: str) -> bool:
        """
        Check if content is duplicate with optimized fingerprint-based approach
        
        Args:
            content: Text content to check
            url: URL of the content
            
        Returns:
            True if content is considered duplicate
        """
        if not content or len(content.strip()) < 100:
            return True
            
        # Calculate hash for exact duplicate detection
        content_hash = self.calculate_content_hash(content)
        
        # Check exact hash match first (fastest)
        if content_hash in self.seen_content_hashes:
            logger.info(f"Exact duplicate found for URL: {url}")
            return True
        
        # Create fingerprint for faster similarity grouping
        fingerprint = self.create_content_fingerprint(content)
        
        # Only check similarity with content that has similar fingerprints
        if fingerprint in self.content_fingerprints:
            for existing_url in self.content_fingerprints[fingerprint]:
                existing_content = self.url_content_map.get(existing_url, "")
                if existing_content and self.calculate_similarity(content, existing_content) > self.similarity_threshold:
                    logger.info(f"Similar content found: {url} is similar to {existing_url} (ratio: {self.calculate_similarity(content, existing_content):.3f})")
                    return True
        
        # If not duplicate, store for future comparison
        self.seen_content_hashes.add(content_hash)
        self.url_content_map[url] = content[:1000]
        self.content_fingerprints[fingerprint].append(url)
        
        return False
    
    def clean_and_deduplicate_data(self, data: List[Dict]) -> List[Dict]:
        """
        Clean and deduplicate entire dataset with optimized batch processing
        
        Args:
            data: List of data items with url, title, text_content fields
            
        Returns:
            Cleaned and deduplicated data
        """
        cleaned_data = []
        processed_urls = set()
        
        logger.info(f"Starting cleaning and deduplication of {len(data)} items")
        
        # Pre-filter obviously invalid items including 404 content
        valid_items = []
        for item in data:
            # Basic validation
            if not item.get('url') or not item.get('text_content'):
                continue
            
            text_content = item.get('text_content', '').strip()
            
            # Check minimum length
            if len(text_content) < 50:
                continue
            
            # Check for 404 Not Found content (case insensitive)
            text_lower = text_content.lower()
            if '404 not found' in text_lower or '404' in text_lower and 'not found' in text_lower:
                logger.info(f"Skipping 404 content for URL: {item.get('url')}")
                continue
            
            # Check for other error pages
            if any(error_phrase in text_lower for error_phrase in [
                'page not found',
                'error 404',
                '404 error',
                'the page you requested was not found',
                'this page could not be found',
                'sorry, the page you are looking for could not be found'
            ]):
                logger.info(f"Skipping error page content for URL: {item.get('url')}")
                continue
            
            valid_items.append(item)
        
        logger.info(f"Pre-filtered to {len(valid_items)} valid items (removed {len(data) - len(valid_items)} invalid/404 items)")
        
        for i, item in enumerate(valid_items):
            try:
                # Clean and validate URL
                original_url = item['url']
                cleaned_url = self.validate_and_fix_url(original_url)
                
                # Skip if URL already processed
                if cleaned_url in processed_urls:
                    continue
                
                # Clean text content
                cleaned_content = self.clean_html_content(item['text_content'])
                
                # Skip if content becomes too short after cleaning
                if len(cleaned_content.strip()) < 100:
                    continue
                
                # Double-check for 404 content after HTML cleaning
                cleaned_lower = cleaned_content.lower()
                if '404 not found' in cleaned_lower or ('404' in cleaned_lower and 'not found' in cleaned_lower):
                    logger.info(f"Skipping 404 content after cleaning for URL: {cleaned_url}")
                    continue
                
                # Check for duplicates
                if self.is_duplicate_content(cleaned_content, cleaned_url):
                    continue
                
                # Create cleaned item
                cleaned_item = {
                    'url': cleaned_url,
                    'title': self.clean_html_content(item.get('title', '')),
                    'text_content': cleaned_content,
                    'source_type': item.get('source_type', 'html'),
                    'timestamp': item.get('timestamp', ''),
                    'content_length': len(cleaned_content),
                    'original_url': original_url if original_url != cleaned_url else None
                }
                
                cleaned_data.append(cleaned_item)
                processed_urls.add(cleaned_url)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(valid_items)} items, kept {len(cleaned_data)}")
                    
            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
                continue
        
        logger.info(f"Cleaning completed: {len(data)} -> {len(cleaned_data)} items")
        return cleaned_data
    
    def get_cleaning_stats(self) -> Dict[str, int]:
        """
        Get statistics about the cleaning process
        
        Returns:
            Dictionary with cleaning statistics
        """
        return {
            'unique_content_hashes': len(self.seen_content_hashes),
            'stored_content_samples': len(self.url_content_map),
            'fingerprint_groups': len(self.content_fingerprints),
            'similarity_threshold': self.similarity_threshold
        }


def clean_data_file(input_file: str, output_file: str, similarity_threshold: float = 0.95) -> Dict[str, int]:
    """
    Clean and deduplicate data file with optimized performance
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output cleaned JSON file
        similarity_threshold: Threshold for content similarity detection (increased default)
        
    Returns:
        Statistics about the cleaning process
    """
    import json
    
    logger.info(f"Loading data from {input_file}")
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} items")
    
    # Clean data
    cleaner = DataCleaner(similarity_threshold=similarity_threshold)
    cleaned_data = cleaner.clean_and_deduplicate_data(data)
    
    # Save cleaned data
    logger.info(f"Saving {len(cleaned_data)} cleaned items to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    stats = {
        'original_count': len(data),
        'cleaned_count': len(cleaned_data),
        'removed_count': len(data) - len(cleaned_data),
        'reduction_percentage': round((1 - len(cleaned_data) / len(data)) * 100, 2) if data else 0
    }
    
    logger.info(f"Data cleaning completed: {stats}")
    return stats


if __name__ == "__main__":
    # Test the data cleaner
    import os
    
    logging.basicConfig(level=logging.INFO)
    
    input_file = "data/output.json"
    output_file = "data/output.json"
    
    if os.path.exists(input_file):
        stats = clean_data_file(input_file, output_file)
        print(f"Cleaning statistics: {stats}")
    else:
        print(f"Input file not found: {input_file}")