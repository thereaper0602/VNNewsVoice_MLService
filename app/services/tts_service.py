from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import feedparser
import time
from time import mktime
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
import boto3
from botocore.exceptions import NoCredentialsError
from app.services.aws_storage_service import S3StorageService

class ArticleTTSService:
    @staticmethod
    def _save_wave_file(filename: str, pcm_data: bytes, channels: int = 1, rate:int = 24000, sample_width:int = 2):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)
            
    @staticmethod
    def test_tts_with_short_text() -> Optional[bytes]:
        short_text = "Đây là bản tin thời sự. Xin chào quý vị và các bạn."
        return ArticleTTSService.generate_tts(short_text, "Zephyr")

    @staticmethod
    def generate_tts(text: str, voice_name: str = "Zephyr") -> Optional[bytes]:
        try:
            env_path = Path(__file__).parent.parent.parent / ".env"
            if env_path.exists():
                from dotenv import load_dotenv
                load_dotenv(env_path, override=True)
            
            api_key = settings.GOOGLE_AI_API_KEY
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
        """Generate TTS and upload to AWS S3 (updated to use S3 instead of Cloudinary)"""
        try:
            # Generate TTS audio data
            audio_data = ArticleTTSService.generate_tts(text, voice_name)
            if not audio_data:
                return None
            
            # Create S3 storage service
            s3_service = S3StorageService()
            
            # Create filename with timestamp
            timestamp = int(time.time())
            filename = f"tts_{timestamp}_{voice_name.lower()}.wav"
            
            # Upload to S3
            upload_result = s3_service.upload_audio(audio_data, filename)
            print(upload_result)
            
            if not upload_result or upload_result.get('status') != 'success':
                print("❌ Upload failed")
                return None
            
            # Format result to match expected schema
            result = {
                "status": "success",
                "audio_url": upload_result["audio_url"],
                "audio_size": len(audio_data),
                "voice_name": voice_name,
                "text_length": len(text),
                "filename": filename,
                "upload_timestamp": upload_result.get("created_at", datetime.now().isoformat()),
                "cloud_provider": "aws_s3",
                "public_id": filename  # Use filename as public_id for S3
            }
            
            return result
            
        except Exception as e:
            print(f"❌ TTS upload error: {e}")
            import traceback
            traceback.print_exc()
            return None

        
    
    @staticmethod
    def generate_tts_with_upload_cloudinary(text: str, voice_name: str = "Zephyr") -> Optional[dict]:
        try:
            # Load environment variables if not already loaded
            env_path = Path(__file__).parent.parent.parent / ".env"
            if env_path.exists():
                from dotenv import load_dotenv
                load_dotenv(env_path, override=True)
                
            # Configure Cloudinary
            cloudinary.config(
                cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
                api_key=os.getenv("CLOUDINARY_API_KEY"),
                api_secret=os.getenv("CLOUDINARY_API_SECRET")
            )
            
            # Log Cloudinary configuration status (without revealing secrets)
            print(f"🔐 Cloudinary Config Check:")
            print(f"   Cloud name: {os.getenv('CLOUDINARY_CLOUD_NAME', 'Not set')}")
            print(f"   API key: {'Set' if os.getenv('CLOUDINARY_API_KEY') else 'Not set'}")
            print(f"   API secret: {'Set' if os.getenv('CLOUDINARY_API_SECRET') else 'Not set'}")
            
            # Generate TTS
            audio_data = ArticleTTSService.generate_tts(text, voice_name)
            if not audio_data:
                return None
            
            # 🔧 CREATE CONSISTENT PUBLIC ID
            timestamp = int(time.time())
            filename = f"tts_{timestamp}_{voice_name.lower()}"
            
            # 🔧 SIMPLER FOLDER STRUCTURE
            folder = "vnnews/audio"  # Single level thay vì nested
            public_id = f"{folder}/{filename}"  # vnnews/audio/tts_1753688075_zephyr
            
            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(
                audio_data,
                public_id=public_id,
                resource_type="raw",  # Important for audio files
                folder=None,  # Don't auto-add folder since we handle in public_id
                overwrite=True
            )
            
            result = {
                "status": "success",
                "audio_url": upload_result['secure_url'],
                "public_id": upload_result['public_id'],  # 🔑 EXACT PUBLIC ID
                "audio_size": len(audio_data),
                "voice_name": voice_name,
                "text_length": len(text),
                "filename": f"{filename}.wav",
                "upload_timestamp": datetime.now().isoformat(),
                "cloud_provider": "cloudinary"
            }
            
            print(f"✅ Upload success:")
            print(f"   URL: {result['audio_url']}")
            print(f"   Public ID: {result['public_id']}")  # 🔑 LOG PUBLIC ID
            
            return result
            
        except Exception as e:
            print(f"❌ TTS upload error: {e}")
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