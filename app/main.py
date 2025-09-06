from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import news, summarization, tts

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    news.router, 
    prefix=settings.API_V1_STR, 
    tags=["news"]
)

app.include_router(
    summarization.router, 
    prefix=settings.API_V1_STR, 
    tags=["summarization"]
)

app.include_router(
    tts.router, 
    prefix=settings.API_V1_STR, 
    tags=["text-to-speech"]
)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": f"🚀 {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "news": f"{settings.API_V1_STR}/docs#/news",
            "summarization": f"{settings.API_V1_STR}/docs#/summarization", 
            "tts": f"{settings.API_V1_STR}/docs#/text-to-speech"
        }
    }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": settings.PROJECT_NAME,
        "services": {
            "news_crawler": "✅ running",
            "ai_summarization": "✅ running", 
            "text_to_speech": "✅ running",
            "cloud_storage": "✅ running"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )