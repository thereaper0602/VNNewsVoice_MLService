# app/schemas/tts.py
from pydantic import BaseModel, Field
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
    public_id: str = Field(..., description="Cloud storage public ID to delete")