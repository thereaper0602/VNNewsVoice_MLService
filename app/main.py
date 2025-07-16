from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import *
from services import NewsService
from typing import List, Optional
import time
from config import settings


app = FastAPI(
    title="Vietnamese News Crawler API",
    description="API for crawling Vietnamese news articles and RSS feeds",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {
        "message": f"🚀 {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs"
    }

@app.post(f"{settings.API_V1_STR}/crawl/article", response_model=APIResponse)
async def crawl_article(request: CrawlArticleRequest):
    """
    Crawl a single news article from URL
    """
    try:
        title, blocks = NewsService.crawl_news_article(str(request.url), request.generator)
        
        if not blocks:
            raise HTTPException(status_code=400, detail="Could not extract content from article")
        
        article = Article(
            title=title,  # 🎯 Dùng title thật từ trang web
            url=str(request.url),
            published_at=None,
            blocks=blocks
        )
        
        return APIResponse(
            success=True,
            data=[article.model_dump()],
            message=f"Successfully crawled article: '{title}' with {len(blocks)} blocks"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error crawling article: {str(e)}")

@app.post(f"{settings.API_V1_STR}/crawl/rss", response_model=APIResponse)
async def crawl_rss_feed(request: CrawlRssRequest):
    """
    Crawl articles from an RSS feed
    """
    try:
        articles = NewsService.get_rss_feed(
            str(request.rss_url),
            max_articles=request.max_articles,
            last_crawl_time=request.last_crawl_time
        )
        
        return APIResponse(
            success=True,
            data=[article.model_dump() for article in articles],
            message=f"Successfully crawled {len(articles)} articles"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error crawling RSS feed: {str(e)}")

@app.get(f"{settings.API_V1_STR}/crawl/vietnamese-news", response_model=APIResponse)
async def crawl_vietnamese_news(max_articles: int = 5):
    """
    Crawl latest news from major Vietnamese news sources
    """
    vietnamese_rss_feeds = {
        "vnexpress": "https://vnexpress.net/rss/tin-moi-nhat.rss",
        "thanhnien": "https://thanhnien.vn/rss/home.rss",
        "dantri": "https://dantri.com.vn/rss/home.rss",
        "tuoitre": "https://tuoitre.vn/rss/tin-moi-nhat.rss"
    }
    
    all_articles = []
    
    for source, rss_url in vietnamese_rss_feeds.items():
        try:
            articles = NewsService.get_rss_feed(rss_url, max_articles)
            all_articles.extend(articles)
            time.sleep(3)  # Rate limiting between sources
        except Exception as e:
            print(f"Error crawling {source}: {e}")
            continue
    
    return APIResponse(
        success=True,
        data=[article.model_dump() for article in all_articles],  # 🔧 Sửa .dict() → .model_dump()
        message=f"Successfully crawled {len(all_articles)} articles from Vietnamese news sources"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)