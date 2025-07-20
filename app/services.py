from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import feedparser
import time
from time import mktime
from models import Article, ArticleBlock, CrawlArticleRequest, APIResponse
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

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")
# model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-large-vietnews-summarization").to(device)

class NewsService:
    @staticmethod
    def parse_datetime_flexible(date_input: Union[str, int, datetime, None]) -> Optional[datetime]:
        """
        Parse datetime từ nhiều định dạng khác nhau
        """
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
    def crawl_news_article(article_url: str, generator: Optional[str] = None) -> tuple[str, List[ArticleBlock]]:
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
                if title and len(title) > 5:  # Ensure it's a real title
                    break

        # 📄 EXTRACT CONTENT
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
            
        # Remove unwanted elements
        for unwanted in container.find_all(['script', 'style', 'iframe', 'ads']):
            unwanted.decompose()
            
        # Remove any unwanted related news sections
        for related_div in container.find_all('div', attrs={'type': 'RelatedOneNews'}):
            related_div.decompose()
            
        blocks = []
        order = 1
        seen_paragraphs = set()
        seen_figcaptions = set()

        for elem in container.find_all(['p', 'h1', 'h2', 'h3', 'figure','picture'], recursive=True):
            block_data = {"order": order}
            
            if elem.name in ['h1', 'h2', 'h3']:
                block_data['type'] = 'heading'
                block_data['text'] = elem.get_text(strip=True)
                block_data['tag'] = elem.name
                
            elif elem.name == 'p':
                if generator and generator.lower() == 'vnexpress' and elem.has_attr('class') and 'Normal' in elem['class']:
                    text = elem.get_text(strip=True)
                    if text and text not in seen_paragraphs and text not in seen_figcaptions:
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
                
        return title, blocks
    
    @staticmethod
    def get_rss_feed(rss_url: str, max_articles: int = 5, last_crawl_time: Optional[datetime] = None) -> List[Article]:
        try:
            feed = feedparser.parse(rss_url)
            if not feed.entries:
                raise ValueError("No entries found in the RSS feed")
                
            articles = []
            generator = feed.get('generator')
            count = 0
            
            # Parse last_crawl_time flexible
            parsed_last_crawl_time = NewsService.parse_datetime_flexible(last_crawl_time)
            
            for entry in feed.entries:
                if count >= max_articles:
                    break
                    
                try:
                    # 🔧 XỬ LÝ PUBLISHED DATE - SỬA LẠI
                    published_date = None
                    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
                    
                    # 🔧 Ưu tiên dùng published string thay vì published_parsed
                    if hasattr(entry, 'published') and entry.published:
                        try:
                            from dateutil import parser
                            # dateutil parse trực tiếp từ string, giữ nguyên timezone
                            published_date = parser.parse(entry.published)
                            print(f"📅 RAW RSS: {entry.published}")
                            print(f"📅 PARSED: {published_date}")
                            
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
                    
                    # Crawl article content
                    title, blocks = NewsService.crawl_news_article(entry.link, generator=generator)
                    if not blocks:
                        print(f"❌ SKIP: No content extracted")
                        continue
                    
                    final_title = entry.title if hasattr(entry, 'title') and entry.title else title
                    
                    article = Article(
                        title=final_title,
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

class ArticleSummarizationService:
    # 🔧 Lazy loading - chỉ load khi cần
    _tokenizer = None
    _model = None
    _device = None
    
    @classmethod
    def _load_model(cls):
        """Lazy load model để tránh chậm khởi động"""
        if cls._model is None:
            print("🤖 Loading ViT5 summarization model...")
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")
            cls._model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base-vietnews-summarization")
            cls._model.to(cls._device)
            cls._model.eval()  # Set to evaluation mode
            print(f"✅ Model loaded on {cls._device}")
    
    @classmethod
    def chunk_text(cls, text: str, max_tokens: int = 512) -> List[str]:
        """
        Chia text thành chunks phù hợp với ViT5
        """
        cls._load_model()
        
        # 🔧 Cải thiện việc split sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Skip sentences quá ngắn hoặc không có nghĩa
            if len(sentence.strip()) < 10:
                continue
                
            test_chunk = (current_chunk + " " + sentence).strip()
            
            # 🔧 Tokenize để check length chính xác
            token_count = len(cls._tokenizer.encode(test_chunk, add_special_tokens=True))
            
            if token_count > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        # Add chunk cuối
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    @classmethod
    def summarize_text(cls, text: str, max_length: int = 150) -> str:
        """
        Tóm tắt text với error handling
        """
        try:
            cls._load_model()
            
            if not text or len(text.strip()) < 50:
                return "Nội dung quá ngắn để tóm tắt"
            
            # 🔧 Clean text trước khi process
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            
            # 🔧 GIẢM CHUNK SIZE để nhanh hơn
            chunks = cls.chunk_text(cleaned_text, max_tokens=256)  # Giảm từ 512 xuống 256
            
            if not chunks:
                return "Không có nội dung để tóm tắt"
            
            summaries = []
            
            for i, chunk in enumerate(chunks):
                try:
                    print(f"📝 Summarizing chunk {i+1}/{len(chunks)}...")
                    
                    # 🔧 Tokenize với proper settings
                    inputs = cls._tokenizer(
                        chunk, 
                        return_tensors="pt", 
                        max_length=256,  # Giảm từ 512
                        truncation=True, 
                        padding=True
                    ).to(cls._device)
                    
                    # 🔧 Generate với FASTER parameters
                    with torch.no_grad():
                        summary_ids = cls._model.generate(
                            inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_length=min(max_length, 100),  # Giảm max_length
                            min_length=20,                    # Giảm min_length
                            num_beams=3,                      # Giảm từ 4 xuống 2
                            length_penalty=1.2,               # Giảm từ 2.0
                            early_stopping=True,
                            no_repeat_ngram_size=2,
                            do_sample=False                   # Deterministic generation
                        )
                    
                    summary = cls._tokenizer.decode(
                        summary_ids[0], 
                        skip_special_tokens=True
                    ).strip()
                    
                    if summary and len(summary) > 10:
                        summaries.append(summary)
                        print(f"✅ Chunk {i+1} summarized: {len(summary)} chars")
                    
                except Exception as e:
                    print(f"❌ Error summarizing chunk {i+1}: {e}")
                    continue
            
            if not summaries:
                return "Không thể tóm tắt nội dung"
            
            # 🔧 Join và clean final summary
            final_summary = " ".join(summaries)
            final_summary = re.sub(r'\s+', ' ', final_summary).strip()
            
            # # Trim nếu quá dài
            # if len(final_summary) > max_length * 1.5:
            #     final_summary = final_summary[:int(max_length * 1.5)] + "..."
            
            print(f"🎯 Final summary: {len(final_summary)} chars")
            return final_summary
            
        except Exception as e:
            print(f"❌ Error in summarization: {e}")
            return "Lỗi khi tóm tắt nội dung"

    @classmethod
    def summarize_article(cls, article: Article, max_length: int = 200) -> str:
        """
        Tóm tắt Article object
        """
        if not article.blocks:
            return "Không có nội dung để tóm tắt"
        
        # 🔧 Extract only paragraph content + filter quality
        paragraphs = []
        for block in article.blocks:
            if (block.type == 'paragraph' and 
                block.content and 
                len(block.content.strip()) > 20):  # Skip very short paragraphs
                paragraphs.append(block.content.strip())
        
        if not paragraphs:
            return "Không có đoạn văn để tóm tắt"
        
        full_text = " ".join(paragraphs)
        
        # 🔧 Add context với title
        if article.title:
            full_text = f"Tiêu đề: {article.title}. Nội dung: {full_text}"
        
        return cls.summarize_text(full_text, max_length)
    
    @classmethod
    def cleanup_model(cls):
        """
        Cleanup model để giải phóng memory
        """
        if cls._model is not None:
            del cls._model
            cls._model = None
        if cls._tokenizer is not None:
            del cls._tokenizer
            cls._tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("🧹 Model cleaned up")

class ArticleTTSService:

    @staticmethod
    def _save_wave_file(filename: str, pcm_data: bytes, channels: int = 1, rate:int = 24000, sample_width:int = 2):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)

    # services.py - Sửa ArticleTTSService.generate_tts
    @staticmethod
    def generate_tts(text: str, voice_name: str = "Zephyr") -> Optional[bytes]:
        try:
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                from dotenv import load_dotenv
                load_dotenv(env_path, override=True)
            
            api_key = os.getenv("GOOGLE_AI_API_KEY")
            if not api_key:
                raise ValueError("API key for Google AI is not set in environment variables")
            
            client = genai.Client(api_key=api_key)
            content = f"""Read aloud in a clear, calm, and professional tone, 
                        suitable for news reading. Maintain a steady pace with 
                        natural pauses at punctuation. Keep the delivery neutral 
                        and objective: {text}"""
            
            print(f"🎵 Generating TTS for {len(text)} characters...")
            
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=content,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                )
            )
            
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                
                if candidate.content and candidate.content.parts:
                    part = candidate.content.parts[0]
                    
                    if hasattr(part, 'inline_data') and part.inline_data:
                        pcm_data = part.inline_data.data
                        print(f"🎵 Raw PCM data: {len(pcm_data)} bytes")
                        
                        if isinstance(pcm_data, bytes):
                            # 🔧 CONVERT PCM TO WAV
                            wav_data = ArticleTTSService._pcm_to_wav(pcm_data)
                            print(f"✅ Converted to WAV: {len(wav_data)} bytes")
                            return wav_data
                        else:
                            print(f"❌ Unexpected data type: {type(pcm_data)}")
                            return None
            
            print("❌ No audio data found")
            return None
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _pcm_to_wav(pcm_data: bytes, channels: int = 1, sample_rate: int = 24000, sample_width: int = 2) -> bytes:
        """
        Convert raw PCM data to WAV format
        """
        import io
        import struct
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        
        wav_buffer.seek(0)
        wav_data = wav_buffer.getvalue()
        
        print(f"🔧 PCM to WAV conversion:")
        print(f"   Input PCM: {len(pcm_data)} bytes")
        print(f"   Output WAV: {len(wav_data)} bytes")
        print(f"   WAV header: {wav_data[:12]}")
        
        return wav_data
