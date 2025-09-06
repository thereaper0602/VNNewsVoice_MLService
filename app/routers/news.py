from fastapi import APIRouter, HTTPException
from typing import List
import time

from app.models.response import APIResponse
from app.schemas.article import CrawlArticleRequest, CrawlRssRequest
from app.models.article import Article
from app.services.crawl_news_service import NewsService

router = APIRouter()

@router.post("/crawl/article", response_model=APIResponse)
async def crawl_article(request: CrawlArticleRequest):
    """Crawl a single news article from URL"""
    try:
        title, blocks = NewsService.crawl_news_article(str(request.url), request.generator)
        
        if not blocks:
            raise HTTPException(status_code=400, detail="Could not extract content from article")
        
        article = Article(
            title=title,
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

@router.post("/crawl/rss", response_model=APIResponse)
async def crawl_rss_feed(request: CrawlRssRequest):
    """Crawl articles from an RSS feed"""
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

@router.get("/crawl/vietnamese-news", response_model=APIResponse)
async def crawl_vietnamese_news(max_articles: int = 5):
    """Lấy tin tức từ các nguồn báo chí Việt Nam phổ biến"""
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
        data=[article.model_dump() for article in all_articles],
        message=f"Successfully crawled {len(all_articles)} articles from Vietnamese news sources"
    )