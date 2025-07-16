from pydantic_settings import BaseSettings  # Đổi import này
from typing import List, Optional
import os

class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "VNNewsVoice ML Service"
    DESCRIPTION: str = "Vietnamese News Crawler and ML Service"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Crawler
    CRAWL_INTERVAL_MINUTES: int = 5
    MAX_ARTICLES_PER_CRAWL: int = 5
    RATE_LIMIT_SECONDS: int = 2  # Giảm xuống 2s cho nhanh hơn

    # Logging
    LOG_LEVEL: str = "INFO"

    # Allowed hosts for CORS
    ALLOWED_HOSTS: List[str] = ["*"]  # Đơn giản hóa

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()