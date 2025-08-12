from typing import Optional
import os
from pathlib import Path
import cloudinary
import cloudinary.uploader
import uuid
from app.core.config import settings

class CloudStorageService:
    @staticmethod
    def _get_config():
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path, override=True)
        
        # cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
        # api_key = os.getenv("CLOUDINARY_API_KEY")
        # api_secret = os.getenv("CLOUDINARY_API_SECRET")

        cloud_name = settings.CLOUDINARY_CLOUD_NAME
        api_key = settings.CLOUDINARY_API_KEY
        api_secret = settings.CLOUDINARY_API_SECRET
        
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
            
            # Đảm bảo filename không có đuôi .wav
            filename_without_ext = filename.replace('.wav', '')
            
            # Tạo public_id đúng cấu trúc - chỉ dùng một cách để chỉ định folder
            # KHÔNG sử dụng cả folder trong public_id và tham số folder
            public_id = f"vnnews/audio/{filename_without_ext}"
            
            print(f"🔼 Uploading audio with public_id: {public_id}")
            
            # Thêm phần mở rộng .wav vào public_id để đảm bảo URL có đuôi file
            public_id_with_ext = f"{public_id}.wav"
            
            print(f"🔼 Uploading audio with public_id: {public_id_with_ext}")
            
            upload_result = cloudinary.uploader.upload(
                audio_data,
                resource_type='raw',    # Dùng 'raw' để giữ nguyên tên file và phần mở rộng
                public_id=public_id,    # Không thêm .wav vào public_id, Cloudinary sẽ giữ nguyên cấu trúc
                overwrite=True,         # Cho phép ghi đè nếu file đã tồn tại  
                use_filename=True,      # Sử dụng tên file gốc
                format="wav"            # Chỉ định rõ định dạng là WAV
            )
            if upload_result and 'secure_url' in upload_result:
                # Lấy URL từ kết quả upload
                secure_url = upload_result['secure_url']
                
                # Đảm bảo URL kết thúc bằng .wav
                if not secure_url.endswith('.wav'):
                    secure_url = f"{secure_url}.wav"
                    
                print(f"🔗 Final URL with extension: {secure_url}")
                
                result = {
                    'audio_url': secure_url,
                    'public_id': upload_result['public_id'],
                    'bytes': upload_result.get('bytes', len(audio_data)),
                    'format': 'wav',    # Cố định format là WAV
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
    def delete_audio(public_id: str, resource_type: str = 'video') -> bool:
        try:
            config = CloudStorageService._get_config()
            cloudinary.config(**config)
            
            print(f"🗑️ Trying to delete: {public_id} (resource_type: {resource_type})")
            
            result = cloudinary.uploader.destroy(
                public_id,
                resource_type=resource_type
            )
            
            if result and result.get('result') == 'ok':
                print(f"✅ Audio deleted: {public_id}")
                return True
            else:
                print(f"❌ Delete failed: {result}")
                
                # Thử lại với resource_type khác nếu lần đầu không thành công
                if resource_type == 'video':
                    print(f"🔄 Retrying with resource_type='raw'")
                    return CloudStorageService.delete_audio(public_id, 'raw')
                
                return False
        except Exception as e:
            print(f"❌ Cloudinary delete error: {e}")
            return False
    
    @staticmethod
    def extract_public_id_from_url(url: str) -> str:
        """Extract public ID từ Cloudinary URL"""
        try:
            # URL format: https://res.cloudinary.com/cloud_name/video/upload/v1234/vnnews/audio/filename.wav
            if "cloudinary.com" not in url:
                raise ValueError("Not a Cloudinary URL")
                
            print(f"🔍 Analyzing URL: {url}")
            
            # Xác định resource_type từ URL (video hoặc raw)
            resource_type = "video"  # Default for audio
            if "/raw/" in url:
                resource_type = "raw"
            elif "/video/" in url:
                resource_type = "video"
            elif "/image/" in url:
                resource_type = "image"
                
            print(f"📂 Detected resource_type: {resource_type}")
            
            # Split URL at upload part
            parts = url.split('/upload/')
            if len(parts) != 2:
                raise ValueError(f"Invalid Cloudinary URL format: {url}")
            
            after_upload = parts[1]
            
            # Remove version (vXXXXX)
            if '/' in after_upload and after_upload.split('/')[0].startswith('v'):
                version_removed = after_upload.split('/', 1)[1]
            else:
                version_removed = after_upload
                
            # Remove file extension (.wav, .mp3, etc)
            if '.' in version_removed:
                public_id = version_removed.rsplit('.', 1)[0]
                print(f"🔧 Removed file extension: {version_removed} → {public_id}")
            else:
                public_id = version_removed
                print(f"🔧 No file extension found in URL")
            
            print(f"🔑 Extracted public ID: {public_id}")
            print(f"🔧 Resource type: {resource_type}")
            
            return public_id
        except Exception as e:
            print(f"❌ Error extracting public ID from URL '{url}': {e}")
            return ""
    
    @staticmethod
    def delete_audio_by_url(audio_url: str) -> bool:
        """Delete audio bằng URL (convenience method)"""
        try:
            public_id = CloudStorageService.extract_public_id_from_url(audio_url)
            if not public_id:
                print("❌ Could not extract public_id from URL")
                return False
            
            # Xác định resource_type từ URL
            resource_type = 'video'  # Default cho audio
            if '/raw/' in audio_url:
                resource_type = 'raw'
            elif '/video/' in audio_url:
                resource_type = 'video'
                
            print(f"🗑️ Deleting from URL with resource_type: {resource_type}")
            
            return CloudStorageService.delete_audio(public_id, resource_type)
            
        except Exception as e:
            print(f"❌ Delete by URL error: {e}")
            return False
        
    @staticmethod
    def search_and_delete_audio(filename_pattern: str) -> bool:
        """Search for audio file then delete it"""
        try:
            config = CloudStorageService._get_config()
            cloudinary.config(**config)
            
            import cloudinary.api
            
            print(f"🔍 Searching for files matching: {filename_pattern}")
            
            # Search in different resource types
            for resource_type in ['raw', 'video']:
                try:
                    # Search by prefix
                    search_results = cloudinary.api.resources(
                        resource_type=resource_type,
                        type="upload",
                        prefix="vnnews",  # Search in vnnews folder
                        max_results=100
                    )
                    
                    files = search_results.get('resources', [])
                    print(f"📊 Found {len(files)} {resource_type} files in vnnews")
                    
                    # Look for matching file
                    for file in files:
                        file_public_id = file.get('public_id', '')
                        file_url = file.get('secure_url', '')
                        
                        # Check if this matches our target
                        if filename_pattern in file_public_id or filename_pattern in file_url:
                            print(f"🎯 FOUND MATCH!")
                            print(f"   Public ID: {file_public_id}")
                            print(f"   URL: {file_url}")
                            print(f"   Resource Type: {resource_type}")
                            
                            # Try to delete this exact match
                            delete_result = cloudinary.uploader.destroy(
                                file_public_id,
                                resource_type=resource_type
                            )
                            
                            print(f"🗑️ Delete result: {delete_result}")
                            
                            if delete_result and delete_result.get('result') == 'ok':
                                print(f"✅ Successfully deleted: {file_public_id}")
                                return True
                            
                except Exception as e:
                    print(f"❌ Search error for {resource_type}: {e}")
                    continue
            
            print(f"❌ No matching files found for: {filename_pattern}")
            return False
            
        except Exception as e:
            print(f"❌ Search and delete error: {e}")
            return False