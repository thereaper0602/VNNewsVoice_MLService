# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import Response
# from app.models_test import *
# from app.services_test import NewsService, ArticleSummarizationService, ArticleTTSService, CloudinaryService
# from typing import List, Optional
# import time
# from app.core.config import settings


# app = FastAPI(
#     title="Vietnamese News Crawler API",
#     description="API for crawling Vietnamese news articles and RSS feeds",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 🔧 Thay đổi từ settings.ALLOWED_HOSTS
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# async def root():
#     return {
#         "message": f"🚀 {settings.PROJECT_NAME}",
#         "version": settings.VERSION,
#         "status": "running",
#         "docs": "/docs"
#     }

# @app.post(f"{settings.API_V1_STR}/crawl/article", response_model=APIResponse)
# async def crawl_article(request: CrawlArticleRequest):
#     """
#     Crawl a single news article from URL
#     """
#     try:
#         title, blocks = NewsService.crawl_news_article(str(request.url), request.generator)
        
#         if not blocks:
#             raise HTTPException(status_code=400, detail="Could not extract content from article")
        
#         article = Article(
#             title=title,  # 🎯 Dùng title thật từ trang web
#             url=str(request.url),
#             published_at=None,
#             blocks=blocks
#         )


        
#         return APIResponse(
#             success=True,
#             data=[article.model_dump()],
#             message=f"Successfully crawled article: '{title}' with {len(blocks)} blocks"
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error crawling article: {str(e)}")

# @app.post(f"{settings.API_V1_STR}/crawl/rss", response_model=APIResponse)
# async def crawl_rss_feed(request: CrawlRssRequest):
#     """
#     Crawl articles from an RSS feed
#     """
#     try:
#         articles = NewsService.get_rss_feed(
#             str(request.rss_url),
#             max_articles=request.max_articles,
#             last_crawl_time=request.last_crawl_time
#         )
        
#         return APIResponse(
#             success=True,
#             data=[article.model_dump() for article in articles],
#             message=f"Successfully crawled {len(articles)} articles"
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error crawling RSS feed: {str(e)}")

# @app.get(f"{settings.API_V1_STR}/crawl/vietnamese-news", response_model=APIResponse)
# async def crawl_vietnamese_news(max_articles: int = 5):
#     """
#     Crawl latest news from major Vietnamese news sources
#     """
#     vietnamese_rss_feeds = {
#         "vnexpress": "https://vnexpress.net/rss/tin-moi-nhat.rss",
#         "thanhnien": "https://thanhnien.vn/rss/home.rss",
#         "dantri": "https://dantri.com.vn/rss/home.rss",
#         "tuoitre": "https://tuoitre.vn/rss/tin-moi-nhat.rss"
#     }
    
#     all_articles = []
    
#     for source, rss_url in vietnamese_rss_feeds.items():
#         try:
#             articles = NewsService.get_rss_feed(rss_url, max_articles)
#             all_articles.extend(articles)
#             time.sleep(3)  # Rate limiting between sources
#         except Exception as e:
#             print(f"Error crawling {source}: {e}")
#             continue
    
#     return APIResponse(
#         success=True,
#         data=[article.model_dump() for article in all_articles],  # 🔧 Sửa .dict() → .model_dump()
#         message=f"Successfully crawled {len(all_articles)} articles from Vietnamese news sources"
#     )

# @app.post(f"{settings.API_V1_STR}/summarize", response_model=APIResponse)
# async def summarize_content(request: SummarizeRequest):
#     """
#     Summarize text content using AI model
#     """
#     try:
#         if not request.content or len(request.content.strip()) < 50:
#             raise HTTPException(status_code=400, detail="Content too short to summarize")
        
#         # Tóm tắt bằng AI
#         summary = ArticleSummarizationService.summarize_text(
#             request.content, 
#             max_length=request.max_length
#         )
        
#         return APIResponse(
#             success=True,
#             data=[{
#                 "original_length": len(request.content),
#                 "summary_length": len(summary),
#                 "summary": summary,
#                 "compression_ratio": round(len(summary) / len(request.content), 2)
#             }],
#             message="Successfully summarized content"
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error summarizing content: {str(e)}")

# @app.post(f"{settings.API_V1_STR}/summarize/article", response_model=APIResponse)
# async def summarize_article_by_url(request: CrawlArticleRequest):
#     """
#     Crawl article and summarize in one step (for convenience)
#     """
#     try:
#         # Crawl article
#         title, blocks = NewsService.crawl_news_article(str(request.url), request.generator)
        
#         if not blocks:
#             raise HTTPException(status_code=400, detail="Could not extract content from article")
        
#         article = Article(
#             title=title,
#             url=str(request.url),
#             published_at=None,
#             blocks=blocks
#         )
        
#         # Tóm tắt
#         summary = ArticleSummarizationService.summarize_article(article)
        
#         return APIResponse(
#             success=True,
#             data=[{
#                 "article": article.model_dump(),
#                 "summary": summary,
#                 "summary_length": len(summary)
#             }],
#             message=f"Successfully crawled and summarized article: '{title}'"
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error processing article: {str(e)}")

# @app.post(f"{settings.API_V1_STR}/tts")
# async def text_to_speech(request : dict):
#     try:
#         text = request.get("text", "").strip()
#         if not text or len(text) < 10:
#             raise HTTPException(status_code=400, detail="Text too short for TTS")
#         if len(text) > 5000:
#             raise HTTPException(status_code=400, detail="Text too long for TTS (max 5000 characters)")
#         voice_name = request.get("voice_name", "Zephyr")
#         audio_data = ArticleTTSService.generate_tts(
#             text,
#             voice_name=voice_name
#         )
#         if audio_data is None:
#             raise HTTPException(status_code=500, detail="Failed to generate TTS audio")
#         return Response(
#             content=audio_data,
#             media_type="audio/wav",
#             headers={
#                 "Content-Disposition": "attachment; filename=speech.wav",
#                 "Content-Length": str(len(audio_data))
#             }
#         )
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")

# @app.post(f"{settings.API_V1_STR}/tts/upload", response_model=APIResponse)
# async def text_to_speech_with_upload(request: dict):
#     """
#     Generate TTS and upload to Cloudinary, return audio URL
#     """
#     try:
#         text = request.get("text", "").strip()
#         if not text or len(text) < 10:
#             raise HTTPException(status_code=400, detail="Text too short for TTS (min 10 chars)")
#         if len(text) > 5000:
#             raise HTTPException(status_code=400, detail="Text too long for TTS (max 5000 chars)")
        
#         voice_name = request.get("voice_name", "Zephyr")
        
#         print(f"🚀 TTS with upload request: {len(text)} chars, voice: {voice_name}")
        
#         # Generate TTS and upload to Cloudinary
#         result = ArticleTTSService.generate_tts_with_upload(text, voice_name)
        
#         if not result:
#             raise HTTPException(status_code=500, detail="Failed to generate TTS or upload to cloud")
        
#         return APIResponse(
#             success=True,
#             data=[result],
#             message=f"TTS generated and uploaded successfully. Audio URL: {result['audio_url']}"
#         )
        
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         print(f"❌ Endpoint error: {e}")
#         raise HTTPException(status_code=500, detail=f"Error processing TTS upload: {str(e)}")

# @app.delete(f"{settings.API_V1_STR}/tts/delete/{{public_id}}")
# async def delete_tts_audio(public_id: str):
#     """
#     Delete audio file from Cloudinary
#     """
#     try:
#         # Decode public_id if URL encoded
#         import urllib.parse
#         decoded_public_id = urllib.parse.unquote(public_id)
        
#         success = CloudinaryService.delete_audio(decoded_public_id)
        
#         if success:
#             return APIResponse(
#                 success=True,
#                 data=[{"public_id": decoded_public_id, "deleted": True}],
#                 message="Audio file deleted successfully"
#             )
#         else:
#             raise HTTPException(status_code=404, detail="Audio file not found or delete failed")
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error deleting audio: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

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