from fastapi import APIRouter, HTTPException

from app.models.response import APIResponse
from app.schemas.article import SummarizeRequest, CrawlArticleRequest
from app.models.article import Article
from app.services.crawl_news_service import NewsService
from app.services.text_summarization import ArticleSummarizationService

router = APIRouter()

@router.post("/summarize", response_model=APIResponse)
async def summarize_content(request: SummarizeRequest):
    """Summarize text content using AI model"""
    try:
        if not request.content or len(request.content.strip()) < 50:
            raise HTTPException(status_code=400, detail="Content too short to summarize")
        
        summary = ArticleSummarizationService.summarize_text(
            request.content, 
            max_length=request.max_length
        )
        
        return APIResponse(
            success=True,
            data=[{
                "original_length": len(request.content),
                "summary_length": len(summary),
                "summary": summary,
                "compression_ratio": round(len(summary) / len(request.content), 2)
            }],
            message="Successfully summarized content"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error summarizing content: {str(e)}")

@router.post("/summarize/article", response_model=APIResponse)
async def summarize_article_by_url(request: CrawlArticleRequest):
    """Crawl article and summarize in one step"""
    try:
        title, top_image, blocks = NewsService.crawl_news_article(str(request.url), request.generator)
        if not blocks:
            raise HTTPException(status_code=400, detail="Could not extract title, top image or content from article")

        article = Article(
            title=title,
            url=str(request.url),
            published_at=None,
            blocks=blocks
        )
        
        summary = ArticleSummarizationService.summarize_article(article)
        
        return APIResponse(
            success=True,
            data=[{
                "article": article.model_dump(),
                "summary": summary,
                "summary_length": len(summary)
            }],
            message=f"Successfully crawled and summarized article: '{title}'"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing article: {str(e)}")