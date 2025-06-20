"""
Data models for the crawler
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PageData:
    """Data structure for page information"""
    url: str
    title: str
    text_content: str
    source_type: str
    timestamp: str
    content_hash: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'url': self.url,
            'title': self.title,
            'text_content': self.text_content,
            'source_type': self.source_type,
            'timestamp': self.timestamp,
            'content_hash': self.content_hash,
            'metadata': self.metadata
        }