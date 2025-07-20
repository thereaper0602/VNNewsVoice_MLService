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

class ArticleTTSService:
    @staticmethod
    def _save_wave_file(filename: str, pcm_data: bytes, channels: int = 1, rate:int = 24000, sample_width:int = 2):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)

    @staticmethod
    def generate_tts(text: str, voice_name: str = "Zephyr") -> Optional[bytes]:
        try:
            env_path = Path(__file__).parent.parent.parent / ".env"
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
    def generate_tts_with_upload(text: str, voice_name: str = "Zephyr") -> Optional[dict]:
        """Generate TTS audio and upload to Cloudinary, return URL and metadata"""
        try:
            print(f"🎵 Starting TTS generation and upload for {len(text)} characters...")
            
            # Step 1: Generate TTS audio
            audio_data = ArticleTTSService.generate_tts(text, voice_name)
            
            if not audio_data:
                print("❌ TTS generation failed")
                return None
            
            # Step 2: Upload to Cloudinary
            filename = f"tts_{int(time.time())}_{voice_name.lower()}.wav"
            from services.cloud_service import CloudStorageService
            upload_result = CloudStorageService.upload_audio(audio_data, filename)
            
            if not upload_result:
                print("❌ Cloud upload failed")
                return None
            
            # Step 3: Return comprehensive result
            result = {
                "status": "success",
                "audio_url": upload_result['audio_url'],
                "public_id": upload_result['public_id'],
                "audio_size": len(audio_data),
                "cloud_size": upload_result['bytes'],
                "voice_name": voice_name,
                "text_length": len(text),
                "filename": filename,
                "format": "wav",
                "upload_timestamp": datetime.now().isoformat(),
                "cloud_provider": "cloudinary",
                "cloudinary_info": {
                    "created_at": upload_result.get('created_at'),
                    "version": upload_result.get('version'),
                    "resource_type": upload_result.get('resource_type')
                }
            }
            
            print(f"🎉 TTS + Upload completed successfully!")
            print(f"🔗 Audio URL: {result['audio_url']}")
            
            return result
            
        except Exception as e:
            print(f"❌ TTS with upload error: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _pcm_to_wav(pcm_data: bytes, channels: int = 1, sample_rate: int = 24000, sample_width: int = 2) -> bytes:
        """Convert raw PCM data to WAV format"""
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