from pydantic import BaseModel, Field, field_validator
from typing import Optional

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000, description="Text to convert to speech")
    voice_name: Optional[str] = Field(default="Zephyr", description="Voice model name")

class TTSResponse(BaseModel):
    audio_url: str
    audio_size: int
    voice_name: str
    text_length: int
    filename: str
    upload_timestamp: str
    cloud_provider: str
    public_id: Optional[str] = None

class TTSDeleteRequest(BaseModel):
    public_id: str = Field(..., description="Cloud storage public ID/filename to delete")

class TTSDeleteByUrlRequest(BaseModel):
    audio_url: str = Field(..., description="Audio URL to delete")
    
    @field_validator('audio_url')
    @classmethod
    def validate_url(cls, v):
        if not v or not v.strip():
            raise ValueError("Audio URL cannot be empty")
        
        # Support both Cloudinary and S3 URLs
        if not any(domain in v for domain in ["cloudinary.com", "amazonaws.com"]):
            raise ValueError("Must be a valid cloud storage URL (Cloudinary or AWS S3)")
        
        return v.strip()