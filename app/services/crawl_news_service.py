from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import feedparser
import time
from time import mktime
# ✅ FIX: Import từ app.models thay vì models
from app.models.article import Article, ArticleBlock
from app.models.response import APIResponse
from typing import List, Optional, Union
import pytz
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from google import genai
from google.genai import types
import wave
from pathlib import Path
import cloudinary
import cloudinary.uploader
import uuid
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

class NewsService:
    @staticmethod
    def parse_datetime_flexible(date_input: Union[str, int, datetime, None]) -> Optional[datetime]:
        """Parse datetime từ nhiều định dạng khác nhau"""
        if not date_input:
            return None
            
        if isinstance(date_input, datetime):
            return date_input
            
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        
        try:
            # Case 1: Timestamp (seconds hoặc milliseconds)
            if isinstance(date_input, (int, float)):
                timestamp = float(date_input)
                # Nếu > 1e10 thì là milliseconds, convert về seconds
                if timestamp > 1e10:
                    timestamp = timestamp / 1000
                return datetime.fromtimestamp(timestamp, tz=vietnam_tz)
            
            # Case 2: String formats
            if isinstance(date_input, str):
                from dateutil import parser
                
                # Try parse với dateutil (handle hầu hết formats)
                parsed_date = parser.parse(date_input)
                
                # Nếu không có timezone, assume là Vietnam time
                if parsed_date.tzinfo is None:
                    return vietnam_tz.localize(parsed_date)
                else:
                    # Convert về Vietnam timezone
                    return parsed_date.astimezone(vietnam_tz)
                    
        except Exception as e:
            print(f"Error parsing datetime '{date_input}': {e}")
            return None
        
        return None

    @staticmethod
    def crawl_news_article(article_url: str, generator: Optional[str] = None) -> tuple[str, str, List[ArticleBlock]]:
        response = requests.get(article_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # 🎯 EXTRACT TITLE
        title = "No title found"
        title_selectors = [
            'h1.title-detail',          # VNExpress
            'h1.title',                 # Generic
            'h1.article-title',         # Some news sites
            'h1.entry-title',           # WordPress sites
            '.article-header h1',       # Alternative
            'header h1',                # Header section
            'h1'                        # Fallback - first h1
        ]
        
        for selector in title_selectors:
            title_tag = soup.select_one(selector)
            if title_tag:
                title = title_tag.get_text(strip=True)
                if title and len(title) > 5:
                    break

        # 📄 Trích xuất nội dung
        container = None
        content_selectors = [
            "div.detail-cmain",         # VNExpress
            "article.fck_detail",       # Some sites
            "div.singular-content",     # WordPress
            ".content",                 # Generic
            "div.ArticleContent",       # Alternative
            "div.article-content"       # Alternative
        ]
        
        for selector in content_selectors:
            found = soup.select_one(selector)
            if found:
                container = found
                break
        
        if not container:
            raise ValueError("Could not find article content container")
            
        # Bỏ các phần thẻ không cần thiết
        for unwanted in container.find_all(['script', 'style', 'iframe', 'ads']):
            unwanted.decompose()

        # Bỏ các phần liên quan đến tin tức không mong muốn
        for related_div in container.find_all('div', attrs={'type': 'RelatedOneNews'}):
            related_div.decompose()
            
        blocks = []
        order = 1
        seen_paragraphs = set()
        seen_figcaptions = set()
        first_image = ""
        is_first_image = True

        for elem in container.find_all(['p', 'h1', 'h2', 'h3', 'figure','picture'], recursive=True):
            block_data = {"order": order}
            
            if elem.name in ['h1', 'h2', 'h3','h4','h5','h6']:
                block_data['type'] = 'heading'
                block_data['text'] = elem.get_text(strip=True)
                block_data['tag'] = elem.name
                
            elif elem.name == 'p':
                if generator and generator.lower() == 'vnexpress' and elem.has_attr('class') and 'Normal' in elem['class']:
                    text = elem.get_text(strip=True)
                    if text and text not in seen_paragraphs and text not in seen_figcaptions: # Kiểm tra xem đoạn văn đã được thấy chưa
                        seen_paragraphs.add(text)
                        block_data['type'] = 'paragraph'
                        block_data['content'] = text
                    else:
                        continue
                else:
                    text = elem.get_text(strip=True)
                    if text and len(text) > 10 and text not in seen_paragraphs and text not in seen_figcaptions:
                        seen_paragraphs.add(text)
                        block_data['type'] = 'paragraph'
                        block_data['content'] = text
                    else:
                        continue
                        
            elif elem.name == 'figure' or elem.name == 'picture':
                img_tag = elem.find('img')
                if img_tag:
                    block_data['type'] = 'image'
                    # Try different src attributes
                    src = None
                    for attr in ['data-src', 'data-original', 'src']:
                        if img_tag.has_attr(attr):
                            src = img_tag[attr]
                            break
                    
                    # Handle srcset
                    if not src and img_tag.has_attr('srcset'):
                        srcset = img_tag['srcset']
                        src = srcset.split()[0] if srcset else ''
                    
                    block_data['src'] = src or ''
                    block_data['alt'] = img_tag.get('alt', '')
                    if is_first_image and block_data['src']:
                        first_image = block_data['src']
                        is_first_image = False
                    
                    # Extract caption
                    caption_tag = elem.find('figcaption') or elem.find('p', class_='Image')
                    if caption_tag:
                        caption = caption_tag.get_text(strip=True)
                        if caption not in seen_figcaptions:
                            seen_figcaptions.add(caption)
                            block_data['caption'] = caption
                        else:
                            block_data['caption'] = ''
                    else:
                        block_data['caption'] = ''
                else:
                    continue  # Skip if no img tag found

            # Add block if it has meaningful content
            if block_data.get('type') and (
                block_data.get('content') or 
                block_data.get('text') or 
                block_data.get('src')
            ):
                blocks.append(ArticleBlock(**block_data))
                order += 1

        return title, first_image, blocks

    @staticmethod
    def get_rss_feed(rss_url: str, max_articles: int = 5, last_crawl_time: Optional[datetime] = None) -> List[Article]:
        try:
            feed = feedparser.parse(rss_url)
            if not feed.entries:
                raise ValueError("No entries found in the RSS feed")
                
            articles = []
            generator = feed.get('generator')  # Lấy thông tin generator từ feed
            count = 0
            
            # Parse last_crawl_time flexible
            parsed_last_crawl_time = NewsService.parse_datetime_flexible(last_crawl_time)
            
            for entry in feed.entries:
                if count >= max_articles:
                    break
                    
                try:
                    # 🔧 XỬ LÝ PUBLISHED DATE
                    published_date = None
                    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
                    
                    # 🔧 Ưu tiên dùng published string thay vì published_parsed
                    if hasattr(entry, 'published') and entry.published:
                        try:
                            from dateutil import parser
                            # dateutil parse trực tiếp từ string, giữ nguyên timezone
                            published_date = parser.parse(entry.published)
                            
                            # Nếu đã có timezone, convert về Vietnam time
                            if published_date.tzinfo is not None:
                                published_date = published_date.astimezone(vietnam_tz)
                            else:
                                # Nếu không có timezone, assume là Vietnam time
                                published_date = vietnam_tz.localize(published_date)
                                
                        except Exception as e:
                            print(f"Error parsing published date string: {e}")
                            # Fallback về published_parsed
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                # 🔧 SỬA: Dùng calendar.timegm thay vì mktime
                                import calendar
                                utc_timestamp = calendar.timegm(entry.published_parsed)  # Convert to UTC timestamp
                                published_date = datetime.fromtimestamp(utc_timestamp, tz=pytz.UTC)
                                published_date = published_date.astimezone(vietnam_tz)
                    
                    # Nếu vẫn không có date, dùng current time
                    if not published_date:
                        published_date = datetime.now(vietnam_tz)
                    
                    # 🔧 DEBUG: In ra để check
                    print(f"📰 Article: {entry.title[:50]}...")
                    print(f"📅 Final Published: {published_date}")
                    print(f"⏰ Last crawl: {parsed_last_crawl_time}")
                    
                    # Logic filter
                    if parsed_last_crawl_time and published_date:
                        if published_date <= parsed_last_crawl_time:
                            print(f"⏭️  SKIP: Article too old ({published_date} <= {parsed_last_crawl_time})")
                            continue
                        else:
                            print(f"✅ INCLUDE: Article is newer ({published_date} > {parsed_last_crawl_time})")
                    
                    # Lấy dữ liệu bài báo
                    title, first_image, blocks = NewsService.crawl_news_article(entry.link, generator=generator)
                    if not blocks:
                        print(f"❌ SKIP: No content extracted")
                        continue
                    
                    final_title = entry.title if hasattr(entry, 'title') and entry.title else title
                    
                    article = Article(
                        title=final_title,
                        top_image=first_image,
                        url=entry.link,
                        published_at=published_date,
                        blocks=blocks
                    )
                    articles.append(article)
                    count += 1
                    print(f"✅ Added article #{count}: {final_title[:50]}...")
                    time.sleep(3)
                    
                except Exception as e:
                    print(f"❌ Error crawling article {entry.link}: {e}")
                    continue
                    
            print(f"📊 FINAL RESULT: {len(articles)} articles crawled")
            return articles
            
        except Exception as e:
            raise ValueError(f"Error parsing RSS feed: {e}")