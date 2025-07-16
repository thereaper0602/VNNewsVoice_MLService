from pydantic import BaseModel, HttpUrl, Field, field_validator
from typing import Optional, List, Union
from datetime import datetime
import pytz

class ArticleBlock(BaseModel):
    order: int
    type: str
    content: Optional[str] = None
    text: Optional[str] = None
    tag: Optional[str] = None
    src: Optional[str] = None
    alt: Optional[str] = None
    caption: Optional[str] = None

class Article(BaseModel):
    title: str
    url: HttpUrl
    published_at: Optional[datetime] = None
    blocks: List[ArticleBlock]
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }

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
    content: str
    max_length: Optional[int] = 200

class APIResponse(BaseModel):
    success: bool
    data: Optional[List[dict]] = None
    message: Optional[str] = None