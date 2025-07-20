from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import feedparser
import time
from time import mktime
# ✅ FIX: Import từ app.models thay vì models
from models.article import Article, ArticleBlock
from models.response import APIResponse
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
        """Tóm tắt text với error handling"""
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
            
            print(f"🎯 Final summary: {len(final_summary)} chars")
            return final_summary
            
        except Exception as e:
            print(f"❌ Error in summarization: {e}")
            return "Lỗi khi tóm tắt nội dung"

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