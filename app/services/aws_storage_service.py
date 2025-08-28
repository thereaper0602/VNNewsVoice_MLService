import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from app.core.config import settings
from typing import Optional
import time
from datetime import datetime
import re

class S3StorageService:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket_name = settings.S3_BUCKET_NAME

    def upload_audio(self, audio_data: bytes, filename: str = None) -> Optional[dict]:
        try:
            if not filename:
                timestamp = int(time.time())
                filename = f"tts_{timestamp}.wav"
            
            # Ensure filename ends with .wav
            if not filename.endswith('.wav'):
                filename = f"{filename}.wav"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=filename,
                Body=audio_data,
                ContentType="audio/wav",
                CacheControl="max-age=86400",
                Metadata = {
                    'uploaded_by': 'tts-service',
                    'created_at': datetime.now().isoformat()
                }
            )
            
            # Generate URL
            audio_url = f"https://{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{filename}"
            
            result = {
                'status': 'success',
                'audio_url': audio_url,
                'filename': filename,
                'bucket': self.bucket_name,
                'bytes': len(audio_data),
                'format': 'wav',
                'created_at': datetime.now().isoformat(),
                'cloud_provider': 'aws_s3'
            }
            
            print(f"✅ Audio uploaded to S3: {audio_url}")
            return result
            
        except NoCredentialsError:
            print("❌ AWS credentials not found")
            return None
        except PartialCredentialsError:
            print("❌ Incomplete AWS credentials")
            return None
        except Exception as e:
            print(f"❌ S3 upload error: {e}")
            return None

    def delete_audio(self, filename: str) -> bool:
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=filename
            )
            print(f"✅ Audio deleted from S3: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ S3 delete error: {e}")
            return False

    def delete_audio_by_url(self, audio_url: str) -> bool:
        try:
            filename = self.extract_filename_from_url(audio_url)
            if not filename:
                print("❌ Could not extract filename from URL")
                return False
            
            return self.delete_audio(filename)
            
        except Exception as e:
            print(f"❌ Delete by URL error: {e}")
            return False

    def extract_filename_from_url(self, url: str) -> str:
        """Extract filename from S3 URL"""
        try:
            if "amazonaws.com" not in url:
                raise ValueError("Not an S3 URL")
            
            # Extract filename from URL
            filename = url.split('/')[-1]
            print(f"🔑 Extracted filename: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error extracting filename from URL '{url}': {e}")
            return ""

    def list_audio_files(self, prefix: str = "") -> list:
        """List all audio files in bucket"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.wav'):
                        files.append({
                            'filename': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'url': f"https://{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{obj['Key']}"
                        })
            
            print(f"📊 Found {len(files)} audio files")
            return files
            
        except Exception as e:
            print(f"❌ Error listing files: {e}")
            return []

    def search_and_delete_audio(self, filename_pattern: str) -> bool:
        """Search for audio file then delete it"""
        try:
            files = self.list_audio_files()
            
            for file in files:
                if filename_pattern in file['filename']:
                    print(f"🎯 FOUND MATCH: {file['filename']}")
                    return self.delete_audio(file['filename'])
            
            print(f"❌ No matching files found for: {filename_pattern}")
            return False
            
        except Exception as e:
            print(f"❌ Search and delete error: {e}")
            return False