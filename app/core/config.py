# app/config.py - Sửa lại hoàn toàn
from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

# Get the parent directory (root of project)
ROOT_DIR = Path(__file__).parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"

class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "VNNewsVoice ML Service"
    DESCRIPTION: str = "Vietnamese News Crawler and ML Service"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Crawler
    CRAWL_INTERVAL_MINUTES: int = 5
    MAX_ARTICLES_PER_CRAWL: int = 5
    RATE_LIMIT_SECONDS: int = 2

    # Logging
    LOG_LEVEL: str = "INFO"

    # AI Keys
    GOOGLE_AI_API_KEY: Optional[str] = None
    GOOGLE_AI_API_KEY_TS: Optional[str] = None

    # Cloudinary configuration
    CLOUDINARY_CLOUD_NAME: Optional[str] = None
    CLOUDINARY_API_KEY: Optional[str] = None
    CLOUDINARY_API_SECRET: Optional[str] = None

    # Huggingface API key
    HUGGINGFACE_API_KEY: Optional[str] = None

    # AWS S3
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: Optional[str] = None
    S3_BUCKET_NAME: Optional[str] = None

    # Allowed hosts for CORS
    ALLOWED_HOSTS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Debug path information
        print(f"📁 Current working dir: {os.getcwd()}")
        print(f"📁 Root project dir: {ROOT_DIR}")
        print(f"📄 .env file path: {ENV_FILE}")
        print(f"📄 .env exists: {ENV_FILE.exists()}")
        
        # Check if API key loaded
        if self.GOOGLE_AI_API_KEY:
            print(f"✅ GOOGLE_AI_API_KEY loaded: {self.GOOGLE_AI_API_KEY[:10]}...{self.GOOGLE_AI_API_KEY[-4:]}")
        else:
            print("❌ GOOGLE_AI_API_KEY not loaded!")

        # Check Cloudinary config
        if all([self.CLOUDINARY_CLOUD_NAME, self.CLOUDINARY_API_KEY, self.CLOUDINARY_API_SECRET]):
            print(f"☁️ Cloudinary config loaded: {self.CLOUDINARY_CLOUD_NAME}")
        else:
            print("⚠️ Cloudinary credentials not found")

        # Check AWS S3 config
        if all([self.AWS_ACCESS_KEY_ID, self.AWS_SECRET_ACCESS_KEY, self.AWS_REGION, self.S3_BUCKET_NAME]):
            print(f"☁️ AWS S3 config loaded: {self.S3_BUCKET_NAME}")
        else:
            print("⚠️ AWS S3 credentials not found")

    class Config:
        env_file = str(ENV_FILE)  # Đường dẫn tuyệt đối tới .env
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()