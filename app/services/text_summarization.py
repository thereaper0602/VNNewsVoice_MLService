import hashlib
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
from app.core.config import settings

class ArticleSummarizationService:
    # 🔧 Lazy loading - chỉ load khi cần
    _tokenizer = None
    _model = None
    _device = None
    _summary_cache = {}
    
    @classmethod
    def _load_model(cls):
        """Lazy load model để tránh chậm khởi động"""
        if cls._model is None:
            print("🤖 Loading ViT5 summarization model...")
            cls._device = torch.device("cpu")
            cls._tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")
            cls._model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base-vietnews-summarization")
            # Load quantized model
            quantization_config = torch.quantization.quantize_dynamic(
                {torch.nn.Linear}, dtype=torch.qint8
            )
            cls._model = AutoModelForSeq2SeqLM.from_pretrained(
                "VietAI/vit5-base-vietnews-summarization",
                device_map="auto",
                quantization_config=quantization_config,
                low_cpu_mem_usage=True
            )
    
    @classmethod
    def chunk_text(cls, text: str, max_tokens: int = 512) -> List[str]:
        """Chia text thành chunks phù hợp với ViT5"""
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
        """Tóm tắt text với API first, model fallback"""
        # Generate cache key để tránh gọi API lặp lại
        cache_key = hashlib.md5(text[:1000].encode()).hexdigest()
        
        # Check cache trước
        if cache_key in cls._summary_cache:
            print("✅ Using cached summary")
            return cls._summary_cache[cache_key]
        
        # Clean text
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        if len(cleaned_text) < 50:
            return "Nội dung quá ngắn để tóm tắt"
        
        # 1. Thử dùng Gemini API
        try:
            print("🤖 Trying summarization with Google Gemini API...")
            env_path = Path(__file__).parent.parent.parent / ".env"
            if env_path.exists():
                from dotenv import load_dotenv
                load_dotenv(env_path, override=True)
                
            api_key = settings.GOOGLE_AI_API_KEY_TS
            if not api_key:
                print("⚠️ GOOGLE_AI_API_KEY_TS not found in environment")
                raise ValueError("Google AI API key not set")
                
            # Giới hạn độ dài văn bản để giảm token
            if len(cleaned_text) > 8000:
                print(f"⚠️ Text too long ({len(cleaned_text)} chars), truncating to 8000 chars")
                cleaned_text = cleaned_text[:8000] + "..."
                
            client = genai.Client(api_key=api_key)
            MODEL_ID = "gemini-1.5-flash"  # Model nhẹ hơn
            
            prompt = f"""Tóm tắt văn bản sau trong khoảng {max_length} từ, giữ lại thông tin quan trọng nhất. 
            Viết tóm tắt ngắn gọn, súc tích, dễ hiểu, đủ ý chính. Văn bản:
            
            {cleaned_text}"""

            print(f"📝 Summarizing text of length {len(cleaned_text)}...")

            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt
            )
            
            summary = response.text.strip()
            if summary:
                print(f"✅ Gemini API summary: {len(summary)} chars")
                # Lưu vào cache
                cls._summary_cache[cache_key] = summary
                return summary
            else:
                raise ValueError("Empty summary returned from API")
                
        except Exception as api_error:
            print(f"⚠️ Gemini API error: {str(api_error)}")
            print("🔄 Falling back to local model...")
            
            # 2. Fallback to local model
            try:
                # Load model if needed
                cls._load_model()
                
                if not cls._model or not cls._tokenizer:
                    raise ValueError("Failed to load local model")
                    
                # Process with local model
                chunks = cls.chunk_text(cleaned_text, max_tokens=256)
                
                if not chunks:
                    raise ValueError("No chunks to summarize")
                
                summaries = []
                
                for i, chunk in enumerate(chunks):
                    try:
                        print(f"📝 Summarizing chunk {i+1}/{len(chunks)} with local model...")
                        
                        inputs = cls._tokenizer(
                            chunk, 
                            return_tensors="pt", 
                            max_length=256,
                            truncation=True, 
                            padding=True
                        ).to(cls._device)
                        
                        # Use lower resource settings
                        with torch.no_grad():
                            summary_ids = cls._model.generate(
                                inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                max_length=min(max_length, 100),
                                min_length=20,
                                num_beams=2,  # Reduced from 3
                                length_penalty=1.0,  # Reduced from 1.2
                                early_stopping=True,
                                no_repeat_ngram_size=2,
                                do_sample=False
                            )
                        
                        summary = cls._tokenizer.decode(
                            summary_ids[0], 
                            skip_special_tokens=True
                        ).strip()
                        
                        if summary and len(summary) > 10:
                            summaries.append(summary)
                        
                    except Exception as chunk_error:
                        print(f"❌ Error summarizing chunk {i+1}: {chunk_error}")
                        continue
                
                if not summaries:
                    raise ValueError("No summaries generated")
                
                final_summary = " ".join(summaries)
                final_summary = re.sub(r'\s+', ' ', final_summary).strip()
                
                print(f"✅ Local model summary: {len(final_summary)} chars")
                # Lưu vào cache
                cls._summary_cache[cache_key] = final_summary
                return final_summary
                
            except Exception as local_error:
                print(f"❌ Local model error: {str(local_error)}")
                
                # 3. Emergency fallback
                try:
                    # Extract first few sentences
                    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
                    valid_sentences = [s for s in sentences if len(s.strip()) > 10][:3]
                    
                    if valid_sentences:
                        emergency_summary = ". ".join(valid_sentences) + "."
                        print(f"⚠️ Using emergency fallback summary: {len(emergency_summary)} chars")
                        # Không cache emergency summary vì chất lượng thấp
                        return emergency_summary
                    else:
                        return "Không thể tóm tắt nội dung"
                except Exception as emergency_error:
                    print(f"❌ Emergency fallback error: {str(emergency_error)}")
                    return "Không thể tóm tắt nội dung"
    
    # Thêm phương thức để quản lý cache
    @classmethod
    def clear_old_cache(cls, max_age_minutes=60):
        """Clear cache entries older than max_age_minutes"""
        if not hasattr(cls, "_cache_timestamps"):
            cls._cache_timestamps = {}
        
        current_time = datetime.now()
        expired_keys = []
        
        for key in cls._summary_cache:
            if key not in cls._cache_timestamps:
                cls._cache_timestamps[key] = current_time
                continue
                
            timestamp = cls._cache_timestamps[key]
            age = current_time - timestamp
            
            if age > timedelta(minutes=max_age_minutes):
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in cls._summary_cache:
                del cls._summary_cache[key]
            if key in cls._cache_timestamps:
                del cls._cache_timestamps[key]
        
        print(f"🧹 Cleared {len(expired_keys)} old cache entries")

    @classmethod
    def summarize_article(cls, article: Article, max_length: int = 200) -> str:
        """Tóm tắt Article object"""
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
        """Cleanup model để giải phóng memory"""
        if cls._model is not None:
            del cls._model
            cls._model = None
        if cls._tokenizer is not None:
            del cls._tokenizer
            cls._tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("🧹 Model cleaned up")