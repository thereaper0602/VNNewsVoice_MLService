import hashlib
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import feedparser
import time
from time import mktime
from app.models.article import Article, ArticleBlock
from app.models.response import APIResponse
from typing import List, Optional, Union, Dict, Any
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
from huggingface_hub import InferenceClient

class ArticleSummarizationService:
    # 🔧 Lazy loading - chỉ load khi cần
    _tokenizer = None
    _model = None
    _device = None
    _summary_cache = {}
    _ner_cache = {}
    _ner_client = None
    
    @classmethod
    def _load_model(cls):
        """Lazy load model để tránh chậm khởi động"""
        if cls._model is None:
            print("🤖 Loading ViT5 summarization model...")
            cls._device = torch.device("cpu")
            cls._tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")
            cls._model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base-vietnews-summarization")
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
    def _get_ner_client(cls):
        """Khởi tạo client cho NER API"""
        if cls._ner_client is None:
            print("🤖 Initializing NER client...")
            # Lấy API key từ settings nếu có
            api_key = getattr(settings, "HUGGINGFACE_API_KEY", "")
            cls._ner_client = InferenceClient(
                provider="hf-inference",
                api_key=api_key
            )
        return cls._ner_client
    
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
    def chunk_text_for_ner(cls, text: str, max_chars: int = 1800) -> List[str]:
        """Chia text thành chunks phù hợp với NER model (max 512 tokens)"""
        # Cắt câu theo ., !, ? và cả dấu kết thúc kiểu tiếng Việt
        parts = re.split(r'([.!?。]+)', text)
        # Ghép lại để không mất dấu chấm câu
        sentences = []
        for i in range(0, len(parts), 2):
            sent = parts[i].strip()
            punct = parts[i+1] if i+1 < len(parts) else ""
            if sent:
                sentences.append((sent + punct).strip())

        chunks = []
        current = ""
        for sent in sentences:
            if current and len(current) + 1 + len(sent) > max_chars:
                chunks.append(current)
                current = sent
            else:
                current = sent if not current else (current + " " + sent)
        if current:
            chunks.append(current)
        return chunks
    
    @classmethod
    def process_ner(cls, text: str) -> List[Dict[str, Any]]:
        """Xử lý NER cho văn bản, hỗ trợ chunking cho văn bản dài"""
        # Check cache trước
        cache_key = hashlib.md5(text[:1000].encode()).hexdigest()
        if cache_key in cls._ner_cache:
            print("✅ Using cached NER results")
            return cls._ner_cache[cache_key]
        
        # Khởi tạo client nếu chưa có
        client = cls._get_ner_client()
        
        # Chia text thành chunks nhỏ hơn để tránh vượt quá giới hạn token
        chunks = cls.chunk_text_for_ner(text)
        all_entities = []
        offset = 0
        
        for i, chunk in enumerate(chunks):
            try:
                print(f"🔍 Processing NER for chunk {i+1}/{len(chunks)}...")
                
                # Gọi API NER
                results = client.token_classification(
                    chunk,
                    model="NlpHUST/ner-vietnamese-electra-base"
                )
                
                # Điều chỉnh vị trí về theo văn bản gốc
                for entity in results:
                    entity["start"] += offset
                    entity["end"] += offset
                    all_entities.append(entity)
                    
            except Exception as e:
                print(f"⚠️ NER error for chunk {i+1}: {str(e)}")
                continue
                
            # Cập nhật offset cho chunk tiếp theo
            offset += len(chunk) + 1  # +1 cho khoảng trắng
        
        # Sắp xếp và gộp các entity liên tiếp cùng loại
        all_entities.sort(key=lambda x: (x.get("start", 0), x.get("end", 0)))
        merged_entities = cls._merge_consecutive_entities(all_entities)
        
        # Lưu vào cache
        cls._ner_cache[cache_key] = merged_entities
        
        return merged_entities
    
    @classmethod
    def _merge_consecutive_entities(cls, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Gộp các entity liên tiếp cùng loại"""
        if not entities:
            return entities
            
        merged = []
        current = entities[0].copy()
        
        for entity in entities[1:]:
            # Nếu entity liền kề và cùng loại, gộp lại
            if (entity.get("start") == current.get("end") and 
                entity.get("entity_group") == current.get("entity_group")):
                current["end"] = entity["end"]
                current["word"] = current.get("word", "") + entity.get("word", "")
                # Lấy score cao hơn
                current["score"] = max(current.get("score", 0), entity.get("score", 0))
            else:
                merged.append(current)
                current = entity.copy()
                
        merged.append(current)
        return merged
    
    @classmethod
    def extract_important_entities(cls, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Trích xuất các entity quan trọng theo loại"""
        entity_groups = {}
        
        # Lọc entity có score cao và phân loại
        for entity in entities:
            if entity.get("score", 0) >= 0.7:  # Chỉ lấy entity có độ tin cậy cao
                group = entity.get("entity_group", "OTHER")
                word = entity.get("word", "").strip()
                
                if group not in entity_groups:
                    entity_groups[group] = []
                    
                # Chỉ thêm nếu chưa có trong danh sách
                if word and word not in entity_groups[group]:
                    entity_groups[group].append(word)
        
        return entity_groups

    @classmethod
    def summarize_text(cls, text: str, max_length: int = 150) -> str:
        """Tóm tắt text với API first, model fallback, tích hợp NER"""
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
        
        # Xử lý NER để trích xuất thông tin quan trọng
        try:
            print("🔍 Extracting named entities...")
            entities = cls.process_ner(cleaned_text)
            important_entities = cls.extract_important_entities(entities)
            
            # Tạo context từ các entity quan trọng
            entity_context = ""
            for group, words in important_entities.items():
                if words:
                    entity_context += f"{group}: {', '.join(words[:5])}. "
            
            print(f"✅ Extracted entities: {entity_context}")
        except Exception as ner_error:
            print(f"⚠️ NER extraction error: {str(ner_error)}")
            entity_context = ""
            important_entities = {}
        
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
            
            # Cải thiện prompt với thông tin NER
            prompt = f"""Ban là một chuyên gia trong việc tóm tắt tin tức. Tóm tắt văn bản sau trong khoảng {max_length} từ, giữ lại thông tin quan trọng nhất.
            
            Đặc biệt chú ý đến các đối tượng quan trọng sau đây trong văn bản:
            {entity_context}

            Nếu không có các đối tượng quan trọng, hãy tóm tắt văn bản theo cách thông thường.

            TUYỆT ĐỐI KHÔNG ĐƯỢC BỊA ĐẶT HAY NÓI BẤT CỨ THỨ GÌ KHÔNG CÓ TRONG BÀI BÁO.

            TẤT CẢ THÔNG TIN ĐỀU PHẢI TỪ TIN TỨC MÀ RA.
            
            KHÔNG ĐƯỢC NÓI BẤT CỨ THỨ GÌ KHÔNG CÓ TRONG BÀI BÁO.
            
            Hãy đảm bảo tóm tắt nhấn mạnh các đối tượng quan trọng này (con người, tổ chức, địa điểm, sự kiện).
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
                        
                        # Thêm entity context vào chunk nếu là chunk đầu tiên
                        if i == 0 and entity_context:
                            chunk = f"Thông tin quan trọng: {entity_context}. {chunk}"
                        
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
                
                # Hậu xử lý để đảm bảo các entity quan trọng được nhấn mạnh
                if important_entities:
                    # Thêm thông tin về các entity quan trọng nếu chưa có trong tóm tắt
                    for group, words in important_entities.items():
                        if group in ["PERSON", "ORGANIZATION", "LOCATION", "MISCELLANEOUS"]:
                            for word in words[:3]:  # Chỉ lấy 3 entity quan trọng nhất mỗi loại
                                if word and len(word) > 1 and word not in final_summary:
                                    # Thêm vào đầu tóm tắt
                                    final_summary = f"{word} ({group}): {final_summary}"
                                    break
                
                print(f"✅ Local model summary: {len(final_summary)} chars")
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
        if not hasattr(cls, "_ner_cache_timestamps"):
            cls._ner_cache_timestamps = {}
        
        current_time = datetime.now()
        expired_summary_keys = []
        expired_ner_keys = []
        
        # Xử lý cache tóm tắt
        for key in cls._summary_cache:
            if key not in cls._cache_timestamps:
                cls._cache_timestamps[key] = current_time
                continue
                
            timestamp = cls._cache_timestamps[key]
            age = current_time - timestamp
            
            if age > timedelta(minutes=max_age_minutes):
                expired_summary_keys.append(key)
        
        # Xử lý cache NER
        for key in cls._ner_cache:
            if key not in cls._ner_cache_timestamps:
                cls._ner_cache_timestamps[key] = current_time
                continue
                
            timestamp = cls._ner_cache_timestamps[key]
            age = current_time - timestamp
            
            if age > timedelta(minutes=max_age_minutes):
                expired_ner_keys.append(key)
        
        # Xóa các cache hết hạn
        for key in expired_summary_keys:
            if key in cls._summary_cache:
                del cls._summary_cache[key]
            if key in cls._cache_timestamps:
                del cls._cache_timestamps[key]
                
        for key in expired_ner_keys:
            if key in cls._ner_cache:
                del cls._ner_cache[key]
            if key in cls._ner_cache_timestamps:
                del cls._ner_cache_timestamps[key]
        
        print(f"🧹 Cleared {len(expired_summary_keys)} old summary cache entries")
        print(f"🧹 Cleared {len(expired_ner_keys)} old NER cache entries")

    @classmethod
    def summarize_article(cls, article: Article, max_length: int = 200) -> str:
        """Tóm tắt Article object với nhấn mạnh các đối tượng quan trọng"""
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
        if cls._ner_client is not None:
            cls._ner_client = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("🧹 Model cleaned up")