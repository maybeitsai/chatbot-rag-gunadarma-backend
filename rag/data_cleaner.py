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

logger = logging.getLogger(__name__)


class DataCleaner:
    """Enhanced data cleaning utilities"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.seen_content_hashes: Set[str] = set()
        self.url_content_map: Dict[str, str] = {}
        
        # Regex patterns for cleaning
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\w\s\-.,!?()[\]{}";:\'/\\@#$%^&*+=<>~`|]')
        self.url_spaces_pattern = re.compile(r'\s+')
        
    def clean_html_content(self, text: str) -> str:
        """
        Remove HTML tags and clean up text content
        
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
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove excessive special characters but keep essential punctuation
        # text = self.special_chars_pattern.sub('', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def validate_and_fix_url(self, url: str) -> str:
        """
        Validate and fix URL formatting
        
        Args:
            url: URL to validate and fix
            
        Returns:
            Properly formatted URL with encoded spaces
        """
        if not url:
            return ""
            
        # Remove extra whitespace        url = url.strip()
        
        # Handle spaces in URLs by encoding them as %20
        if ' ' in url:
            # Simply replace spaces with %20 for simplicity and reliability
            url = url.replace(' ', '%20')
        
        return url
    
    def calculate_content_hash(self, content: str) -> str:
        """
        Calculate hash for content deduplication
        
        Args:
            content: Text content to hash
            
        Returns:
            SHA256 hash of normalized content
        """
        # Normalize content for hashing
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity ratio between 0 and 1
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def is_duplicate_content(self, content: str, url: str) -> bool:
        """
        Check if content is duplicate based on hash and similarity
        
        Args:
            content: Text content to check
            url: URL of the content
            
        Returns:
            True if content is considered duplicate
        """
        if not content or len(content.strip()) < 50:  # Skip very short content
            return True
            
        content_hash = self.calculate_content_hash(content)
        
        # Check exact hash match
        if content_hash in self.seen_content_hashes:
            logger.info(f"Exact duplicate found for URL: {url}")
            return True
        
        # Check similarity with existing content
        for existing_url, existing_content in self.url_content_map.items():
            if self.calculate_similarity(content, existing_content) > self.similarity_threshold:
                logger.info(f"Similar content found: {url} is similar to {existing_url}")
                return True
        
        # If not duplicate, store for future comparison
        self.seen_content_hashes.add(content_hash)
        self.url_content_map[url] = content[:1000]  # Store first 1000 chars for similarity comparison
        
        return False
    
    def clean_and_deduplicate_data(self, data: List[Dict]) -> List[Dict]:
        """
        Clean and deduplicate entire dataset
        
        Args:
            data: List of data items with url, title, text_content fields
            
        Returns:
            Cleaned and deduplicated data
        """
        cleaned_data = []
        processed_urls = set()
        
        logger.info(f"Starting cleaning and deduplication of {len(data)} items")
        
        for i, item in enumerate(data):
            try:
                # Skip if missing required fields
                if not item.get('url') or not item.get('text_content'):
                    logger.warning(f"Skipping item {i}: Missing required fields")
                    continue
                
                # Clean and validate URL
                original_url = item['url']
                cleaned_url = self.validate_and_fix_url(original_url)
                
                # Skip if URL already processed
                if cleaned_url in processed_urls:
                    logger.info(f"Skipping duplicate URL: {cleaned_url}")
                    continue
                
                # Clean text content
                cleaned_content = self.clean_html_content(item['text_content'])
                
                # Skip if content is too short or duplicate
                if len(cleaned_content.strip()) < 50:
                    logger.info(f"Skipping URL with short content: {cleaned_url}")
                    continue
                
                if self.is_duplicate_content(cleaned_content, cleaned_url):
                    logger.info(f"Skipping duplicate content: {cleaned_url}")
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
                    logger.info(f"Processed {i + 1}/{len(data)} items, kept {len(cleaned_data)}")
                    
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
            'similarity_threshold': self.similarity_threshold
        }


def clean_data_file(input_file: str, output_file: str, similarity_threshold: float = 0.85) -> Dict[str, int]:
    """
    Clean and deduplicate data file
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output cleaned JSON file
        similarity_threshold: Threshold for content similarity detection
        
    Returns:
        Statistics about the cleaning process
    """
    import json
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Clean data
    cleaner = DataCleaner(similarity_threshold=similarity_threshold)
    cleaned_data = cleaner.clean_and_deduplicate_data(data)
    
    # Save cleaned data
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
