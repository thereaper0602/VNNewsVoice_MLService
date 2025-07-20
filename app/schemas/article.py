from pydantic import BaseModel, HttpUrl, Field, field_validator
from typing import Optional, Union
from datetime import datetime
import pytz

class CrawlArticleRequest(BaseModel):
    url: HttpUrl
    generator: Optional[str] = None

class CrawlRssRequest(BaseModel):
    rss_url: HttpUrl
    max_articles: int = Field(default=5, ge=1)
    last_crawl_time: Optional[Union[str, int, float, datetime]] = None
    
    @field_validator('last_crawl_time', mode='before')
    @classmethod
    def parse_last_crawl_time(cls, v):
        """Convert string/timestamp to datetime"""
        if v is None:
            return None
            
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        
        try:
            # Timestamp
            if isinstance(v, (int, float)):
                timestamp = float(v)
                if timestamp > 1e10:  # milliseconds
                    timestamp = timestamp / 1000
                return datetime.fromtimestamp(timestamp, tz=vietnam_tz)
            
            # String
            if isinstance(v, str):
                from dateutil import parser
                parsed = parser.parse(v)
                if parsed.tzinfo is None:
                    return vietnam_tz.localize(parsed)
                return parsed.astimezone(vietnam_tz)
                
            # Already datetime
            if isinstance(v, datetime):
                if v.tzinfo is None:
                    return vietnam_tz.localize(v)
                return v.astimezone(vietnam_tz)
                
        except Exception as e:
            print(f"Error parsing last_crawl_time '{v}': {e}")
            return None
        
        return v

class SummarizeRequest(BaseModel):
    content: str = Field(..., min_length=50, description="Text content to summarize")
    max_length: Optional[int] = Field(default=200, ge=50, le=500, description="Maximum summary length")

class SummarizeArticleRequest(BaseModel):
    url: HttpUrl
    generator: Optional[str] = None
    max_summary_length: Optional[int] = Field(default=200, ge=50, le=500)
