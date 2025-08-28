from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import urllib.parse

from app.models.response import APIResponse
from app.schemas.tts import TTSRequest, TTSResponse, TTSDeleteByUrlRequest
from app.services.tts_service import ArticleTTSService
from app.services.cloud_service import CloudStorageService
from app.services.aws_storage_service import S3StorageService

router = APIRouter()

@router.post("/tts", response_class=Response)
async def text_to_speech_direct(request: TTSRequest):
    """Generate TTS and return WAV file directly"""
    try:
        audio_data = ArticleTTSService.generate_tts(
            request.text,
            voice_name=request.voice_name
        )
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate TTS audio")
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=speech_{request.voice_name.lower()}.wav",
                "Content-Length": str(len(audio_data))
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")

@router.post("/tts/upload", response_model=APIResponse)
async def text_to_speech_with_upload(request: TTSRequest):
    """Generate TTS and upload to AWS S3, return URL"""
    try:
        print(f"🚀 TTS with S3 upload: {len(request.text)} chars, voice: {request.voice_name}")
        
        result = ArticleTTSService.generate_tts_with_upload(request.text, request.voice_name)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to generate TTS or upload to S3")
        
        tts_response = TTSResponse(
            audio_url=result["audio_url"],
            audio_size=result["audio_size"], 
            voice_name=result["voice_name"],
            text_length=result["text_length"],
            filename=result["filename"],
            upload_timestamp=result["upload_timestamp"],
            cloud_provider=result.get("cloud_provider", "aws_s3"),
            public_id=result.get("public_id")
        )
        
        return APIResponse(
            success=True,
            data=[tts_response.model_dump()],
            message="TTS generated and uploaded to S3 successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing TTS upload: {str(e)}")

@router.delete("/tts/delete/{filename}")
async def delete_tts_audio(filename: str):
    """Delete audio file from S3 storage"""
    try:
        decoded_filename = urllib.parse.unquote(filename)
        s3_service = S3StorageService()
        success = s3_service.delete_audio(decoded_filename)

        if success:
            return APIResponse(
                success=True,
                data=[{"filename": decoded_filename, "deleted": True}],
                message="Audio file deleted successfully from S3"
            )
        else:
            raise HTTPException(status_code=404, detail="Audio file not found or delete failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting audio: {str(e)}")

# @router.delete("/tts/delete/{public_id}")
# async def delete_tts_audio(public_id: str):
#     """Delete audio file from cloud storage"""
#     try:
#         decoded_public_id = urllib.parse.unquote(public_id)
#         # success = CloudStorageService.delete_audio(decoded_public_id)
#         success = S3StorageService.delete_audio(decoded_public_id)

#         if success:
#             return APIResponse(
#                 success=True,
#                 data=[{"public_id": decoded_public_id, "deleted": True}],
#                 message="Audio file deleted successfully"
#             )
#         else:
#             raise HTTPException(status_code=404, detail="Audio file not found or delete failed")
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error deleting audio: {str(e)}")

@router.get("/tts/voices")
async def get_available_voices():
    """Get list of available TTS voices"""
    voices = [
        {"name": "Zephyr", "gender": "male", "language": "en"},
        {"name": "Aria", "gender": "female", "language": "en"},
        {"name": "Nova", "gender": "male", "language": "en"},
        {"name": "Luna", "gender": "female", "language": "en"}
    ]
    
    return APIResponse(
        success=True,
        data=voices,
        message=f"Found {len(voices)} available voices"
    )


@router.delete("/tts/delete-by-url", response_model=APIResponse)
async def delete_tts_audio_by_url(request: dict):
    """Delete audio file from S3 using URL"""
    try:
        audio_url = request.get("audio_url", "").strip()
        
        if not audio_url:
            raise HTTPException(status_code=400, detail="Audio URL is required")
        
        s3_service = S3StorageService()
        success = s3_service.delete_audio_by_url(audio_url)
        
        if success:
            return APIResponse(
                success=True,
                data=[{
                    "audio_url": audio_url,
                    "deleted": True,
                    "cloud_provider": "aws_s3"
                }],
                message="Audio file deleted successfully from S3"
            )
        else:
            raise HTTPException(status_code=404, detail="Audio file not found or delete failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting audio: {str(e)}")