from typing import Optional
import os
from pathlib import Path
import cloudinary
import cloudinary.uploader
import uuid

class CloudStorageService:
    @staticmethod
    def _get_config():
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path, override=True)
        
        cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
        api_key = os.getenv("CLOUDINARY_API_KEY")
        api_secret = os.getenv("CLOUDINARY_API_SECRET")
        
        if not all([cloud_name, api_key, api_secret]):
            raise ValueError("Cloudinary configuration is not set in environment variables")
        
        return {
            "cloud_name": cloud_name,
            "api_key": api_key,
            "api_secret": api_secret
        }
    
    @staticmethod
    def upload_audio(audio_data: bytes, filename: str) -> Optional[dict]:
        try:
            config = CloudStorageService._get_config()
            cloudinary.config(**config)
            if not filename:
                filename = f"audio_{uuid.uuid4().hex[:8]}.wav"
            
            public_id = f"vnnews/{filename.replace('.wav', '')}"
            
            upload_result = cloudinary.uploader.upload(
                audio_data,
                resource_type='raw',
                public_id=public_id,
                format='wav',
                unique_filename=True,
                overwrite=False,
                folder='vnnews/audio'
            )
            if upload_result and 'secure_url' in upload_result:
                result = {
                    'audio_url': upload_result['secure_url'],
                    'public_id': upload_result['public_id'],
                    'bytes': upload_result.get('bytes', len(audio_data)),
                    'format': upload_result.get('format', 'wav'),
                    'resource_type': upload_result.get('resource_type', 'raw'),
                    'created_at': upload_result.get('created_at'),
                    'version': upload_result.get('version')
                }
                print(f"✅ Audio uploaded: {result['audio_url']}")
                return result
            else:
                print("❌ Upload failed: No secure URL returned")
                return None
        except Exception as e:
            print(f"❌ Cloudinary upload error: {e}")
            return None
    
    @staticmethod
    def delete_audio(public_id: str) -> bool:
        try:
            config = CloudStorageService._get_config()
            cloudinary.config(**config)
            
            result = cloudinary.uploader.destroy(
                public_id,
                resource_type='raw'
            )
            
            if result and result.get('result') == 'ok':
                print(f"✅ Audio deleted: {public_id}")
                return True
            else:
                print(f"❌ Delete failed: {result}")
                return False
        except Exception as e:
            print(f"❌ Cloudinary delete error: {e}")
            return False